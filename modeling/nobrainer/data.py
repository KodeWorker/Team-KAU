import glob
from PIL import Image
import torch.utils.data as data

class NobrainDataset(data.Dataset):
    def __init__(self, image_folder, labels_map, transform):
        super(NobrainDataset).__init__()
        self.image_folder = image_folder
        self.labels_map = labels_map
        self.transform = transform
        self.generate_pairs()
    
    def generate_pairs(self):
        self.image_paths = []
        self.labels = []
        for category in self.labels_map.keys():
            images = glob.glob(self.image_folder + "/" + category + "/*.png")
            label = self.labels_map[category]
            
            self.image_paths += images
            self.labels += [label] * len(images)
            
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        data = Image.open(image_path)
        label = self.labels[index]
        
        if self.transform:
            data = self.transform(data)
        
        return data, label
    
    def __len__(self):
        return len(self.image_paths)