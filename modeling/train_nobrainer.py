from efficientnet_pytorch import EfficientNet
import glob
import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from nobrainer.data import NobrainDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import os
from torch.utils.data import WeightedRandomSampler

def log_loss_summary(loss, step, prefix=""):
    print("epoch {} | {}: {}".format(step + 1, prefix + "loss", np.mean(loss)))

def log_scalar_summary(tag, value, step):
    print("epoch {} | {}: {}".format(step + 1, tag, value))
    
if __name__ == "__main__":
    aug_angle = 15
    aug_scale = 0.05
    width_shift_range = 0.1
    height_shift_range = 0.1    
    shear_range = 0.1
    image_size = 224
    labels_map = {"brain": 0, "nobrain": 1}
    image_folder = r"D:\code\AIdea\data\classifier"
    val_ratio = 0.2
    model_name = "efficientnet-b0"
    learning_rate = 1e-3
    batch_size = 4
    epochs = 50
    pretrained_model_path = None
    weights = "./"
    
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    
    transform = transforms.Compose([
                                    transforms.RandomAffine(degrees=(-aug_angle,aug_angle),
                                                            translate=(width_shift_range, height_shift_range),
                                                            scale=(1-aug_scale, 1+aug_scale),
                                                            shear=shear_range),
                                    transforms.Resize(image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ])

    images = glob.glob(image_folder + "/*/*.png")
    n_val = int(len(images)*val_ratio)
    n_train = len(images) - n_val
    
    dataset = NobrainDataset(image_folder, labels_map, transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    target = np.array(train_set.dataset.labels)[train_set.indices]
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(train_set))
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=sampler)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, sampler=None)
    loaders = {"train": train_loader, "valid": valid_loader}
    
    model = EfficientNet.from_name(model_name, override_params={'num_classes': len(labels_map)})
    
    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    
    loss_train = []
    loss_valid = []
    step = 0
    
    min_loss = float("inf")
    for epoch in range(epochs):
    
        for phase in ["train", "valid"]:
            
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            val_pred, val_true = np.array([]), np.array([])
            for i, data in tqdm(enumerate(loaders[phase])):
                if phase == "train":
                    step += 1
    
                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)
                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(x)
                    
                    logps = model.forward(x)
                    loss = criterion(logps, y_true)
                    
                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred = y_pred.detach().cpu().numpy()
                        y_true = y_true.detach().cpu().numpy()
                        val_pred = np.append(val_pred, np.argmax(y_pred, axis=1))
                        val_true = np.append(val_true, y_true)
                        
                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()
                        
            if phase == "train":
                log_loss_summary(loss_train, epoch)
                loss_train = []
            
            if phase == "valid":
                log_loss_summary(loss_valid, epoch, prefix="val_")
                current_loss = np.mean(loss_valid)
                if current_loss < min_loss:
                    min_loss = current_loss
                    torch.save(model.state_dict(), os.path.join(weights, "nobrainer_best_epoch{}.pt".format(epoch+1)))
                log_scalar_summary("lowest_loss", min_loss, epoch)
                log_scalar_summary("valid acc.", accuracy_score(val_pred, val_true), epoch)
                torch.save(model.state_dict(), os.path.join(weights, "nobrainer_last.pt"))
                loss_valid = []