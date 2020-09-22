import nibabel as nib
from preprocessing import normalization, stack_channels_valid, pad_and_resize, resample
from torchvision import transforms
from pytorch_unet.unet import UNet
import torch
from PIL import Image
from tqdm import trange
import cv2
import numpy as np

def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

def reconstruct_label(five_crops, h, w):
    output = np.zeros((h, w))
    crop_h, crop_w = five_crops.shape[1], five_crops.shape[2]
    tl, tr, bl, br, c = five_crops
    
    output[:crop_h,:crop_w] += tl
    output[:crop_h,-crop_w:] += tr
    output[-crop_h:,:crop_w] += bl
    output[-crop_h:,-crop_w:] += br
    
    center_top = int((h-crop_h+1)*0.5)
    center_left = int((w-crop_w+1)*0.5)
    output[center_top:center_top+crop_h,center_left:center_left+crop_w] = output[center_top:center_top+crop_h,center_left:center_left+crop_w]*0.5 + c*0.5
    
    return output

def predict(image, out_file):
    
    in_channels = 3
    out_channels = 1
    init_features = 64
    origin_image_size = 512
    image_size = 256
    labels_map = {"brain": 0, "nobrain": 1}
    unet_model_path = "../weights/unet64/unet_last.pt"
    
    preprocess=[resample, stack_channels_valid, normalization, pad_and_resize]
    
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    #device = torch.device("cpu")
    
    unet_transform = transforms.Compose([
                                    transforms.FiveCrop(image_size),
                                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
                                    ])
    
    
    unet = UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
    unet.to(device)
    unet.load_state_dict(torch.load(unet_model_path))
    unet.eval()
    
    epi_image = nib.load(image)
    epi_image_data = epi_image.get_fdata()
    dummy = epi_image_data.copy()
    h, w = epi_image_data.shape[0], epi_image_data.shape[1]
    
    pixdims = (epi_image.header["pixdim"], epi_image.header["pixdim"])
    for preprocess_ in preprocess:
        if preprocess_ == pad_and_resize:
            epi_image_data, dummy = preprocess_(epi_image_data, dummy, image_size=origin_image_size)
        elif preprocess_ == resample:
            epi_image_data, dummy = preprocess_(epi_image_data, dummy, pixdims = pixdims)
        else:
            epi_image_data, dummy = preprocess_(epi_image_data, dummy)
    
    epi_label_pred = []
    
    # stage two
    for n_slice in trange(epi_image_data.shape[-1]):
     
        input_image = epi_image_data[..., n_slice]
    
        x = unet_transform(Image.fromarray(input_image.transpose(1, 0, 2)))
        
        y_pred = unet(x.to(device))
        y_pred_np = torch.squeeze(y_pred).detach().cpu().numpy()
        
        y_pred_np = reconstruct_label(y_pred_np, h, w)
        y_pred_np = y_pred_np.transpose(1, 0)
        
        y_pred_np = np.round(y_pred_np).astype(np.uint8)
            
        epi_label_pred.append(np.expand_dims(y_pred_np, axis=-1))
        
    epi_label_pred = np.concatenate(epi_label_pred, axis=-1)
    
    img = nib.Nifti1Image(epi_label_pred, affine=None)
    img.to_filename(out_file)