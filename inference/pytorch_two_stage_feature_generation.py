from efficientnet_pytorch import EfficientNet
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
    
def predict(image, out_prob, out_label=None):
    
    in_channels = 3
    out_channels = 1
    init_features = 32
    image_size = 224
    labels_map = {"brain": 0, "nobrain": 1}
    unet_model_path = "./pytorch_unet_models/224-2d-n4-unet32-21.pt"
    efficientnet_model_path = "./pytorch_efficientnet_models/nobrainer.pt"
    model_name = "efficientnet-b0"
    preprocess=[resample, stack_channels_valid, normalization, pad_and_resize]
    
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    #device = torch.device("cpu")
    
    transform = transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    ])
    
    efn = EfficientNet.from_name(model_name, override_params={'num_classes': len(labels_map)})
    efn.load_state_dict(torch.load(efficientnet_model_path))
    efn.to(device)
    
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
            epi_image_data, dummy = preprocess_(epi_image_data, dummy, image_size=image_size)
        elif preprocess_ == resample:
            epi_image_data, dummy = preprocess_(epi_image_data, dummy, pixdims = pixdims)
        else:
            epi_image_data, dummy = preprocess_(epi_image_data, dummy)
    
    epi_label_pred = []
    no_brain = []
    
    # stage one
    #for n_slice in trange(epi_image_data.shape[-1]):
    for n_slice in range(epi_image_data.shape[-1]):
        input_image = epi_image_data[..., n_slice]
        
        x = transform(Image.fromarray(input_image.transpose(1, 0, 2)))
        
        x = torch.unsqueeze(x, 0)
        
        y_nobrain = efn(x.to(device))
        y_nobrain_np = torch.squeeze(y_nobrain).detach().cpu().numpy()
        
        no_brain.append(np.argmax(softmax(y_nobrain_np)))
    
    # expand from center
    n_center_slice = int(epi_image_data.shape[-1]/2)
    for n_slice in range(1, n_center_slice):
        prev, current = no_brain[n_slice-1], no_brain[n_slice]
        if current == labels_map["brain"] and prev == labels_map["nobrain"]:
            no_brain[n_slice:n_center_slice] = [labels_map["brain"]] * (n_center_slice-n_slice)
            break
        
    for n_slice in range(n_center_slice, epi_image_data.shape[-1]):
        prev, current = no_brain[n_slice-1], no_brain[n_slice]
        if current == labels_map["nobrain"] and prev == labels_map["brain"]:
            no_brain[n_slice:epi_image_data.shape[-1]] = [labels_map["nobrain"]] * (epi_image_data.shape[-1]-n_slice)
            break
    
    # stage two
    #for n_slice in trange(epi_image_data.shape[-1]):
    for n_slice in range(epi_image_data.shape[-1]):
     
        if no_brain[n_slice] == labels_map["brain"]:
        
            input_image = epi_image_data[..., n_slice]
        
            x = transform(Image.fromarray(input_image.transpose(1, 0, 2)))
            
            x = torch.unsqueeze(x, 0)
    
            y_pred = unet(x.to(device))
            y_pred_np = torch.squeeze(y_pred).detach().cpu().numpy()
            y_pred_np = y_pred_np.transpose(1, 0)
            
            y_pred_np = cv2.resize(y_pred_np, (h, w), cv2.INTER_CUBIC)
            #y_pred_np = np.round(y_pred_np).astype(np.uint8)
            
        elif no_brain[n_slice] == labels_map["nobrain"]:
        
            #y_pred_np = np.zeros((h, w), dtype=np.uint8)
            y_pred_np = np.zeros((h, w))
            
        epi_label_pred.append(np.expand_dims(y_pred_np, axis=-1))
        
    epi_label_prob = np.concatenate(epi_label_pred, axis=-1)
    epi_label_prediction = np.round(epi_label_prob)
    
    #print("nobrain count: {:d}".format(np.sum(no_brain)))
    img = nib.Nifti1Image(epi_label_prob, affine=None)
    img.to_filename(out_prob)
    
    img = nib.Nifti1Image(epi_label_prediction, affine=None)
    img.to_filename(out_label)