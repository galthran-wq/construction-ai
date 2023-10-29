import os
import argparse

import torch.nn as nn
import numpy as np
import torch
from collections import defaultdict
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from train_copy import get_dataset


def build_args():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-m', '--model', default=None)      
    parser.add_argument('-p', '--preprocessor', default=None)      
    parser.add_argument('-d', '--dataset')      
    parser.add_argument('--height', default=256, type=int)      
    parser.add_argument('-w', default=256, type=int)      
    parser.add_argument('-o', '--output-path', default="./segmentation_example")      
    parser.add_argument('-idx', default=0, type=int)      
    return parser


def draw_panoptic_segmentation(segmentation, segments, model, ax):
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    ax.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments:
        segment_id = segment['id']
        segment_label_id = segment['label_id']

        # todo
        segment_label = model.config.id2label[segment_label_id]

        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))
        
    ax.legend(handles=handles)
    return fig, axs


def preprocess_image(image):
    image_arr = np.array(image)
    transformed_image = train_transform(image=image_arr)["image"]
    inputs = processor(images=[transformed_image], return_tensors="pt")
    return inputs 


def eval(model, inputs):
    if model is not None:
        model = AutoModelForSemanticSegmentation.from_pretrained(model)
    # else: # if you want to compare to before-uptrain result
    #     model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco",
    #                                                       id2label=id2class_general,
    #                                                       ignore_mismatched_sizes=True)
    print(inputs.keys())
    labels = inputs['labels']
    # inputs['pixel_mask'] = inputs['pixel_mask']
    outputs = model(pixel_values=inputs['pixel_values'][None, :, :, :])
    upsampled_logits = nn.functional.interpolate(
        outputs.logits,
        size=labels.shape[::-1],
        mode="bilinear",
        align_corners=False,
    )
    result = upsampled_logits.argmax(dim=1)[0]

    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    # class_queries_logits = outputs.class_queries_logits
    # masks_queries_logits = outputs.masks_queries_logits

    return result
    
# def main(input_path, output_path, model)


if __name__ == "__main__":
    args = build_args().parse_args()
    train_transform = A.Compose([
        A.Resize(width=args.w, height=args.height),
        A.Normalize(mean=[0,0,0], std=[1,1,1]),
    ])
    processor = AutoImageProcessor.from_pretrained(
        args.preprocessor, do_resize=False, do_rescale=False, do_normalize=False, ignore_index=0
    )
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(transform=train_transform, processor=processor)
    idx = args.idx
    inputs = dataset[idx]
    result = eval(args.model, inputs)
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].imshow(dataset.open_as_array(idx))
    axs[1].imshow(result)
    im = axs[2].imshow(inputs['labels'])
    axs[0].set_title("original image")
    axs[1].set_title("predicted segmentation")
    axs[2].set_title("true segmentation")
    # get the colors of the values, according to the 
    # colormap used by imshow
    values = dataset.ID2CLASS
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=values[i]) ) for i in range(len(values)) ]
    axs[2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    fig.savefig(args.output_path)

