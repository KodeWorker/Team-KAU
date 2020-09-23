import os
import glob
import torch.utils.data as data
import numpy as np

class Resize(object):
    def __init__(self, image_size):
        self.image_size = image_size
        
    def __call__(self, x):
        new_x = [np.expand_dims(cv2.resize(x_, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC), axis=0) for x_ in x]
        new_x = np.concatenate(new_x, axis=0)
        return new_x

class EnsembleDataset(data.Dataset):

    def __init__(self, data_dir, transform):
        super(EnsembleDataset).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.init_dataset()
    
    def init_dataset(self):
        self.images = glob.glob(os.path.join(self.data_dir, "images", "*.npy"))
        self.labels = glob.glob(os.path.join(self.data_dir, "labels", "*.npy"))
        
    def __getitem__(self, index):
        
        x = np.load(self.images[index])
        y = np.load(self.labels[index])
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x, y
        
    def __len__(self):
        return len(self.images)