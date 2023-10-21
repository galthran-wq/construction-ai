from tqdm.auto import tqdm
import torch

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = ([example["class_labels"] for example in batch])
    mask_labels = torch.stack([example["mask_labels"] for example in batch])
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}


def get_dataloader(df, *args, **kwargs):
    return torch.utils.data.DataLoader(df, *args, collate_fn=collate_fn, **kwargs)


def compute_mean_std(ds, *args, **kwargs):
    image_loader = get_dataloader(ds, *args, **kwargs)
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    h, w = None, None

    for inputs in tqdm(image_loader):
        # prevent numerical overflow
        image = inputs["pixel_values"] / 255.
        if h is None and w is None:
            _, _, h, w = image.shape
        psum    += image.sum(axis        = [0, 2, 3])
        psum_sq += (image ** 2).sum(axis = [0, 2, 3])

    count = len(ds) * h * w

    # mean and std
    # restore
    total_mean = psum / count * 255
    total_var  = (psum_sq / count * (255 ** 2)) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    return total_mean, total_std

