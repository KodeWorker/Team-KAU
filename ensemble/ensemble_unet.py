import os
import glob 

if __name__ == "__main__":
    
    image_size = 256
    pretrained_model_path = None    
    out_channels = 1
    init_features = 64
    train_feature_dir = "./predictions(validation)"
    
    num_models = len(os.listdir(train_feature_dir))
    
    in_channels = num_models
    unet = UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
    if pretrained_model_path:
        unet.load_state_dict(torch.load(pretrained_model_path))
    unet.to(device)
    
    dataset = EnsembleDataset(train_feature_dir, image_size, tansform=transform)
    