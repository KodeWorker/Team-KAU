from torchvision import datasets, transforms
from pytorch_unet.data import SegmentationDataset
from pytorch_unet.unet import UNet
from pytorch_unet.metrics import DiceLoss
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser()
    # model settings
    parser.add_argument("-b", "--batch_size", help="", default=8, type=int)
    parser.add_argument("-e", "--epoches", help="", default=50, type=int)
    parser.add_argument("-l", "--learning_rate", help="", default=1e-4, type=float)
    parser.add_argument("-w", "--weights", help="", default="./", type=str)
    parser.add_argument("-s", "--image_size", help="", default=320, type=int)
    parser.add_argument("-i", "--in_channels", help="", default=3, type=int)
    parser.add_argument("-o", "--out_channels", help="", default=1, type=int)
    parser.add_argument("-f", "--init_features", help="", default=32, type=int)
    
    # data
    parser.add_argument("-t", "--train_folder", help="", required=True, type=str)
    parser.add_argument("-v", "--validation_folder", help="", required=True, type=str)
    
    # image augmentation
    parser.add_argument("--aug_scale", help="", default=0.05, type=float)
    parser.add_argument("--aug_angle", help="", default=15, type=float)
    parser.add_argument("--width_shift", help="", default=0.1, type=float)
    parser.add_argument("--height_shift", help="", default=0.1, type=float)
    parser.add_argument("--shear", help="", default=0.1, type=float)
    return parser
    
def log_loss_summary(loss, step, prefix=""):
    print("epoch {} | {}: {}".format(step + 1, prefix + "loss", np.mean(loss)))

def log_scalar_summary(tag, value, step):
    print("epoch {} | {}: {}".format(step + 1, tag, value))

#!
def dsc(y_pred, y_true, smooth=1):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    return (np.sum((y_pred * y_true)) * 2.0 + smooth) / (np.sum(y_pred) + np.sum(y_true)+ smooth)
    
if __name__ == "__main__":
    args = build_argparser().parse_args()
    
    batch_size = args.batch_size
    epochs = args.epoches
    lr = args.learning_rate
    weights = args.weights
    image_size = args.image_size
    aug_scale = args.aug_scale
    aug_angle = args.aug_angle
    width_shift_range = args.width_shift
    height_shift_range = args.height_shift
    shear_range = args.shear
    in_channels = args.in_channels
    out_channels = args.out_channels
    init_features = args.init_features
    
    train_folder_path = args.train_folder
    valid_folder_path = args.validation_folder

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    #device = torch.device("cpu")
    tra_transforms = transforms.Compose([
                                    transforms.RandomAffine(degrees=(-aug_angle,aug_angle),
                                                            translate=(width_shift_range, height_shift_range),
                                                            scale=(1-aug_scale, 1+aug_scale),
                                                            shear=shear_range),
                                    transforms.Resize(image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ])
    
    val_transforms = transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        ])
                                    
    train_dataset = SegmentationDataset(train_folder_path, transform=tra_transforms)
    valid_dataset = SegmentationDataset(valid_folder_path, transform=val_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    loaders = {"train": train_loader, "valid": valid_loader}
    
    
    unet = UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
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
                    torch.save(unet.state_dict(), os.path.join(weights, "unet_best_epoch{}.pt".format(epoch+1)))
                log_scalar_summary("best_dsc", best_validation_dsc, epoch)
                torch.save(unet.state_dict(), os.path.join(weights, "unet_last.pt"))
                loss_valid = []
    
    print("\nBest validation mean DSC: {:4f}\n".format(best_validation_dsc))
    
    #state_dict = torch.load(os.path.join(weights, "unet.pt"))
    #unet.load_state_dict(state_dict)
    #unet.eval()