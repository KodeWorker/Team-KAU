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

if __name__ == "__main__":
    
    #filename = "633TNGNB.nii.gz"
    #filename = "A4GY5J5E.nii.gz"
    #filename = "IYO2C7Z7.nii.gz"
    filename = "JMAI55EV.nii.gz"
    image = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\validate_image\image\{}".format(filename)
    
    in_channels = 3
    out_channels = 1
    init_features = 32
    image_size = 224
    labels_map = {"brain": 0, "nobrain": 1}
    unet_model_path = "./pytorch_unet_models/unet_tumors_only2.pt"
    efficientnet_model_path = "./pytorch_efficientnet_models/nobrainer_prototype2.pt"
    model_name = "efficientnet-b0"
    preprocess=[resample, stack_channels_valid, normalization, pad_and_resize]
    
    #device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    device = torch.device("cpu")
    
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
    nobrain_count = 0
    for n_slice in range(epi_image_data.shape[-1]-10, epi_image_data.shape[-1]):
        input_image = epi_image_data[..., n_slice]
        
        x = transform(Image.fromarray(input_image.transpose(1, 0, 2)))
        
        x = torch.unsqueeze(x, 0)
        
        y_nobrain = efn(x.to(device))
        y_nobrain_np = torch.squeeze(y_nobrain).detach().cpu().numpy()
        y_pred = softmax(y_nobrain_np)
        
        print(y_pred)