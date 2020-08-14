import os
from PIL import Image
import glob
import torch
import torch.utils.data as data
import numpy as np
import random

class SegmentationDataset(data.Dataset):
    def __init__(self, folder_path, transform):
        super(SegmentationDataset).__init__()
        self.transform = transform
        self.img_files = glob.glob(os.path.join(folder_path,'imgs','*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_path,'masks',os.path.basename(img_path)))
        
    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = Image.open(img_path)
        label = Image.open(mask_path)
        
        if self.transform:
        
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            img = self.transform(data)
            
            random.seed(seed) # apply this seed to target tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            msk = self.transform(label)
        else:
            img = torch.from_numpy(data).float()
            msk = torch.from_numpy(label).float()
            
        return img, msk

    def __len__(self):
        return len(self.img_files)