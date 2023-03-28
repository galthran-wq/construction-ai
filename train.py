from facade_datasets.const import id2class_general, WINDOW_ID
from transformers import MaskFormerForInstanceSegmentation
import evaluate

# Replace the head of the pre-trained model
# We specify ignore_mismatched_sizes=True to replace the already fine-tuned classification head by a new one
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco",
                                                          id2label=id2class_general,
                                                          ignore_mismatched_sizes=True)

# %%
import albumentations as A
import numpy as np
from transformers import MaskFormerImageProcessor
from facade_datasets.etrims.dataset import EtrimsDataset
from facade_datasets.oxford_cars.dataset import CarsTrainDataset, CarsTestDataset
from facade_datasets.cmp.dataset import CMPDataset

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# note that you can include more fancy data augmentation methods here
train_transform = A.Compose([
    A.Resize(width=256, height=256),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])
processor = MaskFormerImageProcessor(do_resize=False, do_rescale=False, do_normalize=False, ignore_index=0)


etrims = EtrimsDataset(transform=train_transform, processor=processor)
cmp = CMPDataset(transform=train_transform, processor=processor)
cars_train = CarsTrainDataset(transform=train_transform, processor=processor) 
cars_test = CarsTestDataset(transform=train_transform, processor=processor) 

# %%
import torch

etrims_generator = torch.Generator().manual_seed(42)
cars_generator = torch.Generator().manual_seed(42)
cmp_generator = torch.Generator().manual_seed(42)

etrims_train, etrims_val = torch.utils.data.random_split(
    etrims, [0.8, 0.2],
    generator=etrims_generator
)
cmp_train, cmp_val = torch.utils.data.random_split(
    cmp, [0.8, 0.2],
    generator=cmp_generator
)
cars_train, cars_val = torch.utils.data.random_split(
    cars_train, [0.8, 0.2],
    generator=cars_generator
)

# %%
len(etrims_train), len(etrims_val), len(cars_train), len(cars_val)

# %%
from torch.utils.data import ConcatDataset
train = ConcatDataset([torch.utils.data.Subset(etrims_train, [0]), torch.utils.data.Subset(cmp_train,[0]), torch.utils.data.Subset(cars_train, [0])])
val = ConcatDataset([etrims_val, cmp_val, cars_val])

# %%
len(train), len(val)

# %%
cars_test[0]

# %%
from facade_datasets.utils import collate_fn
from transformers import TrainingArguments
from train_utils import CustomTrainer
import torch.nn as nn

metric = evaluate.load("mean_iou")
accuracy = evaluate.load("accuracy")

def get_mask_pred(output, labels):
    """
    Elements from ```MaskFormerLoss```
    Converts masks logits to actual predicted masks.
    TODO: I know this is ugly; especially having to refer to the model object
    """
    class_queries_logits = output.class_queries_logits
    masks_queries_logits = output.masks_queries_logits
    # (batch_size, example, ...label)
    mask_labels, class_labels = labels

    mask_labels = [torch.tensor(mask_label, dtype=torch.float32) for mask_label in mask_labels]
    class_labels = [torch.tensor(class_label, dtype=torch.int64) for class_label in class_labels]
    
    # this part is taken from ```forward()```
    matcher = model.criterion.matcher
    # retrieve the matching between the outputs of the last layer and the labels
    indices = matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
    # compute the average number of target masks for normalization purposes
    num_masks = model.criterion.get_num_masks(class_labels, device=class_labels[0].device)

    # this part is taken from ```loss_masks()```
    src_idx = model.criterion._get_predictions_permutation_indices(indices)
    tgt_idx = model.criterion._get_targets_permutation_indices(indices)
    pred_masks = masks_queries_logits[src_idx]
    target_masks, _ = model.criterion._pad_images_to_max_in_batch(mask_labels)
    target_masks = target_masks[tgt_idx]
    # upsample predictions to the target size, we have to add one dim to use interpolate
    pred_masks = nn.functional.interpolate(
        pred_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
    ).squeeze(1)

    # this part is taken from ```loss_labels()```
    pred_logits = class_queries_logits
    batch_size, num_queries, _ = pred_logits.shape
    idx = model.criterion._get_predictions_permutation_indices(indices)
    # shape = (batch_size, num_queries)
    target_classes_o = torch.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
    # shape = (batch_size, num_queries)
    target_classes = torch.full(
        (batch_size, num_queries), fill_value=model.criterion.num_labels, dtype=torch.int64, device=pred_logits.device
    )
    target_classes[idx] = target_classes_o
    # target_classes is a (batch_size, num_labels, num_queries), we need to permute pred_logits "b q c -> b c q"
    pred_classes = pred_logits.softmax(dim=-1).argmax(dim=-1)

    # shape (batch_size * num_queries, height, width)
    return (
        (pred_masks.sigmoid() > 0.5).int(), 
        target_masks, 
        pred_classes, 
        target_classes
    )


def compute_metrics(eval_pred):
    from transformers.models.maskformer.modeling_maskformer import MaskFormerModelOutput
    """
    The problem is: MaskFormer doesn't quite output binary mask which can instantly be used to comput IoU.
    I have to use some utilities from the corresponding loss class to obtain the actual mask predictions.
    Unfortunately, this quickly becomes ugly -- I have to:
        pack the logits,
        convert the logits and labels back to tensors,
        refer to the model object,
        copy paste code from different parts of ```MaskFormerCriterion``` 
    to get what I want.
    
    Besides, the evaluation loop has to be rewritten (in an ugly way) for this function to work, for the labels
    that we expect are of shape (total_n_queries, h, w) (binary masks for each query for each image), 
    whereas in the original implementation they are truncated to (batch_size, ...)
    """
    logits, labels = eval_pred
    n = len(labels[1][0])
    keys = ["class_queries_logits", "masks_queries_logits", ]
    logits_dict = { key: torch.Tensor(value) for key, value in zip(keys, logits) }
    logits_out = MaskFormerModelOutput(logits_dict)
    pred_masks, target_masks, pred_class, target_class = get_mask_pred(logits_out, labels)
    result = metric.compute(
        predictions=pred_masks, 
        references=target_masks, 
        num_labels=2, 
        ignore_index=255
    )

    return {
        "mean_iou": result["mean_iou"],
        # TODO: for some reason pred clases still have n_quiries 100
        "window_accuracy": accuracy.compute(
            predictions=(pred_class == WINDOW_ID).int().flatten(),
            references=(target_class == WINDOW_ID).int().flatten(),
        )["accuracy"],
        "n_missing_windows": (
            (pred_class == WINDOW_ID) ^ ((pred_class == WINDOW_ID) * (target_class == WINDOW_ID)) +
            (target_class == WINDOW_ID) ^ ((pred_class == WINDOW_ID) * (target_class == WINDOW_ID))
        ).sum(),
        "accuracy": accuracy.compute(
            predictions=pred_class.flatten(),
            references=target_class.flatten(),
        )["accuracy"]
    }

training_args = TrainingArguments(
    # logging_dir="logs",
    output_dir="test_trainer__overfit_batch2",
    # learning_rate=5e-4,
    save_strategy="steps",
    save_steps=1500,
    # evaluation_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=10,
    dataloader_pin_memory=False,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    num_train_epochs=1000,
    eval_accumulation_steps=10,
    lr_scheduler_type="constant"
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=val,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

# # %%
# trainer.evaluate(val)

# %%
trainer.train()

# %%
# _--------------------------------

# %%
# from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
# from PIL import Image
# import requests

# # load MaskFormer fine-tuned on COCO panoptic segmentation
# image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
# model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# inputs = image_processor(images=image, return_tensors="pt")

# outputs = model(**inputs)
# # model predicts class_queries_logits of shape `(batch_size, num_queries)`
# # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
# class_queries_logits = outputs.class_queries_logits
# masks_queries_logits = outputs.masks_queries_logits

# # you can pass them to image_processor for postprocessing
# result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

# # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
# predicted_panoptic_map = result["segmentation"]
# list(predicted_panoptic_map.shape)

# # %%
# result["segments_info"]


