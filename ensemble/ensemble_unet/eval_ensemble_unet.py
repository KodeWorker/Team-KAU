import os
import glob
from tqdm import tqdm
from pytorch_unet.unet import UNet
import torch
import nibabel as nib
import numpy as np
import cv2 

if __name__ == "__main__":
    
    feature_dir = "../predictions(test)"
    image_size = 256
    out_channels = 1
    init_features = 64
    weights = "../../weights/ensemble_unet64.pt"
    output_dir = "./results/unet64"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    score = {
             "2-stage-224-2d-unet32": 0.5475687 * 0.2,
             "2-stage-256-2d-unet64": 0.5316030 * 0.2,
             "2-stage-256-2d-rc-unet64": 0.4642530 * 0.2,
             "2-stage-256-2d-n4rc-unet64": 0.4782162 * 0.2,
             "2-stage-224-2d-n4-unet32": 0.4700288 * 0.2,
             
             "128-3d-unet16": 0.5915776 * 0.5,
             "128-3d-unet16-pretrained": 0.6051299 * 0.5
            }
    
    model_list = os.listdir(feature_dir)
    
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    
    in_channels = len(model_list)
    unet = UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
    unet.load_state_dict(torch.load(weights))
    unet.to(device, dtype=torch.float)
    unet.eval()
    
    filenames = glob.glob(os.path.join(feature_dir, model_list[0], "*.nii.gz"))
    filenames = [os.path.basename(filename) for filename in filenames]
    
    for filename in tqdm(filenames):
        
        feature_paths = {}
        for model in model_list:
            feature_paths[model] = os.path.join(feature_dir, model, filename)

        features = {key: nib.load(value).get_fdata() for key, value in feature_paths.items()}
        
        y_pred = []
        
        h, w = list(features.values())[0].shape[0], list(features.values())[0].shape[1]
        
        for n_slice in range(list(features.values())[0].shape[-1]):
            
            feature_list = []
            for model, cube in features.items():
                #feature_list.append(np.expand_dims(cv2.resize(cube[..., n_slice]*score[model]/np.sum(list(score.values())), (image_size, image_size), cv2.INTER_CUBIC), axis=0))
                feature_list.append(np.expand_dims(cv2.resize(cube[..., n_slice]*score[model], (image_size, image_size), cv2.INTER_CUBIC), axis=0))
            
            x = torch.from_numpy(np.concatenate(feature_list, axis=0))
            x = torch.unsqueeze(x, 0)
            
            y = unet(x.to(device, dtype=torch.float))
            y_np = torch.squeeze(y).detach().cpu().numpy()
            y_np = cv2.resize(y_np, (h, w), cv2.INTER_CUBIC)
            #y_np = np.round(y_np).astype(np.int)
            y_pred.append(np.expand_dims(y_np, axis=-1))
        
        y_pred = np.concatenate(y_pred, axis=-1)
        y_pred = np.round(y_pred)
        #y_pred =  (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
        
        #print(np.sum(y_pred >= 0.5))
        out_file = os.path.join(output_dir, filename)
        img = nib.Nifti1Image(y_pred, affine=None)
        img.to_filename(out_file)
        
        #break
            