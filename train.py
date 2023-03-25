# %%
from facade_datasets.const import id2class_general
from transformers import MaskFormerForInstanceSegmentation

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

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# note that you can include more fancy data augmentation methods here
train_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])
processor = MaskFormerImageProcessor(do_resize=False, do_rescale=False, do_normalize=False, ignore_index=0)


etrims = EtrimsDataset(transform=train_transform, processor=processor)
cars_train = CarsTrainDataset(transform=train_transform, processor=processor) 
cars_test = CarsTestDataset(transform=train_transform, processor=processor) 

# %%
import torch

etrims_generator = torch.Generator().manual_seed(42)
cars_generator = torch.Generator().manual_seed(42)

etrims_train, etrims_val = torch.utils.data.random_split(
    etrims, [0.8, 0.2],
    generator=etrims_generator
)
cars_train, cars_val = torch.utils.data.random_split(
    cars_train, [0.8, 0.2],
    generator=cars_generator
)

# %%
len(etrims_train), len(etrims_val), len(cars_train), len(cars_val)

# %%
from torch.utils.data import ConcatDataset
train = ConcatDataset([etrims_train, cars_train])
val = ConcatDataset([etrims_val, cars_val])

# %%
len(train), len(val)

# %%
cars_test[0]

# %%
from facade_datasets.utils import collate_fn
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="steps",
    eval_steps=100,
    dataloader_pin_memory=False,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    num_train_epochs=20,
    load_best_model_at_end=True
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs)
        #
        loss = loss[0]
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=val,
    data_collator=collate_fn,
)

# # %%
# trainer.evaluate(val)

# %%
trainer.train("./test_trainer/checkpoint-3500")

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


