import os
import argparse
import numpy as np
import torch
from collections import defaultdict
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation

from facade_datasets.const import class2id_general
from facade_datasets.etrims.const import images_base_path as cmp_images_base_path


def build_args():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-m', '--model')      
    parser.add_argument('-i', '--input-path', default=cmp_images_base_path / os.listdir(cmp_images_base_path)[0])      
    parser.add_argument('-o', '--output-path', default="./segmentation_example")      
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
    # note that you can include more fancy data augmentation methods here
    
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

    train_transform = A.Compose([
        A.Resize(width=512, height=512),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    processor = MaskFormerImageProcessor(do_resize=False, do_rescale=False, do_normalize=False, ignore_index=0)
    image_arr = np.array(image)
    transformed_image = train_transform(image=image_arr)["image"]
    inputs = processor(images=transformed_image, return_tensors="pt")
    return inputs 


def eval(model, inputs):
    model = MaskFormerForInstanceSegmentation.from_pretrained(model)
    model
    outputs = model(**inputs)
    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    # class_queries_logits = outputs.class_queries_logits
    # masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to image_processor for postprocessing
    processor = MaskFormerImageProcessor(do_resize=False, do_rescale=False, do_normalize=False, ignore_index=0)
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return model, result
    
# def main(input_path, output_path, model)


if __name__ == "__main__":
    args = build_args().parse_args()
    image = Image.open(args.input_path)
    preprocessed_image = preprocess_image(image)
    model, result = eval(args.model, preprocessed_image)
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    draw_panoptic_segmentation(result['segmentation'], result['segments_info'], model = model, ax = axs[1])
    axs[0].imshow(image)
    fig.savefig(args.output_path)

