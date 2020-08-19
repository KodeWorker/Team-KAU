import torch
from torchvision import transforms
from pytorch_unet.data import SegmentationDataset
from pytorch_unet.unet import UNet
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

if __name__ == "__main__":
    valid_folder_path = "../data/temp/valid"
    fig_dir = "./fig/demo"
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    image_size = 224
    
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    
    transforms = transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    ])
    valid_dataset = SegmentationDataset(valid_folder_path, transform=transforms)
    
    unet = UNet(in_channels=3, out_channels=1)
    unet.to(device)
    
    unet.load_state_dict(torch.load("unet.pt"))
    unet.eval()
    
    count = 0
    #for x,y in tqdm(valid_dataset):
    for x,y in valid_dataset:
        x = torch.unsqueeze(x, 0)
        y_pred = unet(x.to(device))
        y_pred_np = torch.squeeze(y_pred).detach().cpu().numpy()
        y_true_np = torch.squeeze(y).detach().cpu().numpy()
        
        x_np = torch.squeeze(x).detach().cpu().numpy()
        #print(np.unique(y_true_np), (np.min(y_pred_np), np.max(y_pred_np)))
        x_np = x_np.transpose(1,2,0)
        #print(x_np.shape)
        
        y_pred_np[y_pred_np < 0.5] = 0
        y_pred_np[y_pred_np >= 0.5] = 1
        
        fig, ax = plt.subplots(1,2, figsize=(8,8))
        
        ax[0].imshow(y_true_np, alpha=0.8, cmap=plt.cm.get_cmap("Greens"))
        ax[0].imshow(x_np, alpha=0.3)
        
        ax[1].imshow(y_pred_np, alpha=0.8, cmap=plt.cm.get_cmap("Reds"))
        ax[1].imshow(x_np, alpha=0.3)
        
        filename = os.path.join(fig_dir, "{:04d}.jpg".format(count))
        fig.savefig(filename)
        plt.close(fig)
        
        count += 1
        #break