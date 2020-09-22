import os
import glob
import torch
from pytorch_unet.unet import UNet
from data import EnsembleDataset, Resize
from torchvision import transforms
from pytorch_unet.metrics import DiceLoss
import torch.optim as optim
from tqdm import tqdm

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
    
    pretrained_model_path = None    
    out_channels = 1
    init_features = 64
    data_dir = "../data/ensemble/train"
    epochs = 50
    val_ratio = 0.1
    batch_size = 8
    lr = 1e-4
    num_models = 7
    
    print("Number of base models: {}".format(num_models))
    
    print("Prepare datasets ...")
    transform = None
    dataset = EnsembleDataset(data_dir, transform=transform)
    
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(dataset, [len(dataset)-int(len(dataset)*val_ratio), int(len(dataset)*val_ratio)])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    loaders = {"train": train_loader, "valid": valid_loader}
    
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    
    in_channels = num_models
    unet = UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
    if pretrained_model_path:
        unet.load_state_dict(torch.load(pretrained_model_path))
    unet.to(device, dtype=torch.float)
    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0
    
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    
    loss_train = []
    loss_valid = []
    
    step = 0
    
    print("Start training ...")
    for epoch in range(epochs):
        for phase in ["train", "valid"]:
            
            if phase == "train":
                unet.train()
            else:
                unet.eval()
            
            validation_pred = []
            validation_true = []
    
            for data in tqdm(loaders[phase]):
                if phase == "train":
                    step += 1
    
                x, y_true = data
                x, y_true = x.to(device, dtype=torch.float), y_true.to(device, dtype=torch.float)
                
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