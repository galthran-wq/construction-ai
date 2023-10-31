
# Experiments

## Datasets

- Cloud

https://www.kaggle.com/datasets/sorour/95cloud-cloud-segmentation-on-satellite-images


- LoveDA

https://github.com/Junjue-Wang/LoveDA

- FloodNet

https://github.com/BinaLab/FloodNet-Supervised_v1.0

Download the datasets by following instructions inside each repository



## Models

1. Segformer

nvidia/mit-b0

2. DETR

https://huggingface.co/facebook/detr-resnet-50-panoptic

facebook/detr-resnet-50-panoptic

3. BeiT

https://huggingface.co/microsoft/beit-base-finetuned-ade-640-640
microsoft/beit-base-finetuned-ade-640-640


4. DPT

https://huggingface.co/Intel/dpt-large-ade
Intel/dpt-large-ade


- Cloud

5. Maskformer

facebook/maskformer-swin-large-ade
https://huggingface.co/facebook/maskformer-swin-large-ade


## Run

Options for DATASET: cloud, floodnet, loveda

```sh
export DATASET=cloud
```

Train
```sh
python train_copy.py --checkpoint Intel/dpt-large-ade -d $DATASET --lr 5e-5 -o dpt_loveda -h 256 -w 256
```

Eval
```sh
python visualize_eval.py --model ./dpt_cloud/checkpoint-10180/ -d $DATASET --preprocessor Intel/dpt-large-ade -idx 56 -o seg
```


