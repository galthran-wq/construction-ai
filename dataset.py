import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class CloudDataset(Dataset):
    ID2CLASS = {
        0: "background",
        1: "cloud"
    }

    def __init__(self, processor=None, transform=None, base_path=None):
        super().__init__()
        if base_path is None:
            base_path = Path("./geo_datasets/95-cloud_training_only_additional_to38-cloud")
        r_dir, g_dir, b_dir, nir_dir, gt_dir = (
            base_path / 'train_red_additional_to38cloud', 
            base_path / 'train_green_additional_to38cloud', 
            base_path / 'train_blue_additional_to38cloud', 
            base_path / 'train_nir_additional_to38cloud',
            base_path / 'train_gt_additional_to38cloud'
        )
        self.transform = transform
        self.processor = processor
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        self.pytorch = True
        
    def combine_files(self, r_file: Path, g_dir, b_dir,nir_dir, gt_dir):
        files = {'red': r_file, 
                 'green':g_dir/r_file.name.replace('red', 'green'),
                 'blue': b_dir/r_file.name.replace('red', 'blue'), 
                 'nir': nir_dir/r_file.name.replace('red', 'nir'),
                 'gt': gt_dir/r_file.name.replace('red', 'gt')}
        
        return files
                                       
    def __len__(self):
        return len(self.files)
     
    def open_as_array(self, idx, invert=False, include_nir=False, false_color_aug=False):
        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                           ])
        
        if (false_color_aug):
            indexes = np.arange(3)
            np.random.shuffle(indexes)
            raw_rgb = np.stack([raw_rgb[indexes[0]],
                                raw_rgb[indexes[1]],
                                raw_rgb[indexes[2]],
                               ], axis=2)
        else:
            raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                    np.array(Image.open(self.files[idx]['green'])),
                    np.array(Image.open(self.files[idx]['blue'])),
                   ], axis=2)
    
        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)
    
        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))
    
        # normalize
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask==255, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        x = self.open_as_array(idx, invert=False, include_nir=False, false_color_aug=True)
        y = self.open_mask(idx, add_dims=False)
        y[0][0] = 1

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=x, mask=y)
            image, class_id_map = transformed['image'], transformed['mask']

        inputs = self.processor([image], [class_id_map], return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}
        # for k,v in inputs.items():
        #   inputs[k][0].squeeze_() # remove batch dimension
        if inputs['mask_labels'].shape[0] == 2:
            print(1)

        return inputs
    
    def open_as_pil(self, idx):
        arr = 256*self.open_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s