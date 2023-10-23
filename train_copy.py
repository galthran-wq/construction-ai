import os
import argparse

import albumentations as A
import evaluate
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoImageProcessor, MaskFormerImageProcessor
from transformers.models.maskformer.modeling_maskformer import MaskFormerModelOutput
from transformers import AutoModelForSemanticSegmentation

from dataset import CloudDataset, LoveDADataset, FloodNet
from train_utils import CustomTrainer
from facade_datasets.utils import collate_fn
from facade_datasets.utils import compute_mean_std

metric = evaluate.load("mean_iou")
accuracy = evaluate.load("accuracy")


def get_dataset(dataset_str: str):
    if dataset_str == "cloud":
        return CloudDataset
    elif dataset_str == "loveda":
        return LoveDADataset
    elif dataset_str == "floodnet":
        return FloodNet


def prepare_compute_metrics(num_labels, processor: MaskFormerImageProcessor):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        pred_labels = logits_tensor.detach().cpu().numpy()

        result = metric.compute(
            predictions=pred_labels, 
            references=labels, 
            num_labels=num_labels, 
            ignore_index=255
        )
        return {
            "mean_iou": result["mean_iou"],
            "accuracy": result['overall_accuracy']
            # "accuracy": accuracy.compute(
            #     predictions=predicted_semantic_map.flatten(),
            #     references=labels[0].flatten(),
            # )["accuracy"]
        }
    return compute_metrics

def train(
    checkpoint,
    dataset_str,
    output_dir=None,
    resume_from_checkpoint=None,
    lr=1e-4,
    h=256,
    w=256,
):
    # checkpoint = "facebook/maskformer-swin-large-coco"
    # dataset_str = "cloud"
    Dataset = get_dataset(dataset_str)
    num_labels = len(Dataset.ID2CLASS.keys())
    # Replace the head of the pre-trained model
    # We specify ignore_mismatched_sizes=True to replace the already fine-tuned classification head by a new one
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint,
                                                            id2label=Dataset.ID2CLASS,
                                                            ignore_mismatched_sizes=True)


    # note that you can include more fancy data augmentation methods here
    transform = A.Compose([
        A.Resize(width=w, height=h),
        A.Normalize(mean=[0,0,0], std=[1,1,1]),
    ])
    processor = AutoImageProcessor.from_pretrained(
        checkpoint, 
        do_resize=False, 
        do_rescale=False, 
        do_normalize=False, 
        ignore_index=0
    )
    dataset = Dataset(transform=transform, processor=processor)


    generator = torch.Generator().manual_seed(42)
    train, val = torch.utils.data.random_split(
        dataset, [0.8, 0.2],
        generator=generator
    )

    # mean, std = compute_mean_std(train, batch_size=8)
    # mean /= 255.
    # std /= 255.
    # print(mean, std)
    # transform[1].mean = mean
    # transform[1].std = std

    training_args = TrainingArguments(
        logging_dir="logs",
        output_dir=output_dir,
        learning_rate=lr,
        save_strategy="epoch",
        # save_strategy="steps",
        # save_steps=10,
        # evaluation_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=50,
        dataloader_pin_memory=False,
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        eval_accumulation_steps=16,
        num_train_epochs=10,
        lr_scheduler_type="constant"
    )
    print(training_args.learning_rate)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collate_fn,
        compute_metrics=prepare_compute_metrics(num_labels, processor)
    )

    trainer.train(resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-c', '--checkpoint')      
    parser.add_argument('-d', '--dataset')      
    parser.add_argument('-o', '--output-dir')      
    parser.add_argument('-r', '--resume', default=None)      
    parser.add_argument("--height", default=256, type=int)      
    parser.add_argument('-w', "--width", default=256, type=int)      
    parser.add_argument('--lr', default=1e-4, type=float)      
    args = parser.parse_args()
    train(
        checkpoint=args.checkpoint,
        dataset_str=args.dataset,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume,
        lr=args.lr,
        h=args.height,
        w=args.width,
    )


