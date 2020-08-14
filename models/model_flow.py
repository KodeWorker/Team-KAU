from torchvision import datasets, transforms
from pytorch_unet.data import SegmentationDataset
from pytorch_unet.unet import UNet
from pytorch_unet.metrics import DiceLoss
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os

def log_loss_summary(loss, step, prefix=""):
    print("epoch {} | {}: {}".format(step + 1, prefix + "loss", np.mean(loss)))

def log_scalar_summary(tag, value, step):
    print("epoch {} | {}: {}".format(step + 1, tag, value))

def dsc(y_pred, y_true):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))
    
if __name__ == "__main__":

    batch_size = 8#16
    epochs = 50
    lr = 0.0001
    workers = 2
    weights = "./"
    image_size = 224#512
    aug_scale = 0.05
    aug_angle = 15
    width_shift_range = 0.1
    height_shift_range = 0.1
    shear_range = 0.1
    train_folder_path = "../data/temp/train"
    valid_folder_path = "../data/temp/valid"

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    #device = torch.device("cpu")
    transforms = transforms.Compose([
                                    transforms.RandomAffine(degrees=(-aug_angle,aug_angle),
                                                            translate=(width_shift_range, height_shift_range),
                                                            scale=(1-aug_scale, 1+aug_scale),
                                                            shear=shear_range),
                                    transforms.Resize(image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ])
    
    train_dataset = SegmentationDataset(train_folder_path, transform=transforms)
    valid_dataset = SegmentationDataset(valid_folder_path, transform=transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    loaders = {"train": train_loader, "valid": valid_loader}
    
    
    unet = UNet(in_channels=3, out_channels=1)
    unet.to(device)
    
    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0
    
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    
    loss_train = []
    loss_valid = []
    
    step = 0
    
    for epoch in range(epochs):
        for phase in ["train", "valid"]:
            
            if phase == "train":
                unet.train()
            else:
                unet.eval()
            
            validation_pred = []
            validation_true = []
    
            for i, data in tqdm(enumerate(loaders[phase])):
                if phase == "train":
                    step += 1
    
                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)
                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)
    
                    loss = dsc_loss(y_pred, y_true)
    
                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        
                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()
            
            if phase == "train":
                log_loss_summary(loss_train, epoch)
                loss_train = []

            if phase == "valid":
                log_loss_summary(loss_valid, epoch, prefix="val_")
                mean_dsc = np.mean(
                    dsc(validation_pred, validation_true)
                )
                log_scalar_summary("val_dsc", mean_dsc, epoch)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(weights, "unet.pt"))
                loss_valid = []
    
    print("\nBest validation mean DSC: {:4f}\n".format(best_validation_dsc))
    
    state_dict = torch.load(os.path.join(weights, "unet.pt"))
    unet.load_state_dict(state_dict)
    unet.eval()