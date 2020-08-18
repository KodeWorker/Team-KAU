import nibabel as nib
from preprocessing import normalization, stack_channels_valid, pad_and_resize, resample
from torchvision import transforms
from pytorch_unet.unet import UNet
import torch
from PIL import Image

def predict(image, out_file):
    
    in_channels = 3
    out_channels = 1
    init_features = 32
    image_size = 224
    model_path = "./pytorch_unet_models/prototype_unet.pt"
    preprocess=[resample, stack_channels_valid, normalization, pad_and_resize]
    
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    
    transform = transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    ])
    
    unet = UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
    unet.to(device)
    unet.load_state_dict(torch.load(model_path))
    unet.eval()
    
    epi_image = nib.load(image)
    epi_image_data = epi_image.get_fdata()
    dummy = epi_image_data.copy()
    
    pixdims = (epi_image.header["pixdim"], epi_image.header["pixdim"])
    for preprocess_ in preprocess:
        if preprocess_ == pad_and_resize:
            epi_image_data, dummy = preprocess_(epi_image_data, dummy, image_size=image_size)
        elif preprocess_ == resample:
            epi_image_data, dummy = preprocess_(epi_image_data, dummy, pixdims = pixdims)
        else:
            epi_image_data, dummy = preprocess_(epi_image_data, dummy)
    
    for n_slice in range(epi_image_data.shape[-1]):
        input_image = epi_image_data[..., n_slice]
        x = transform(Image.fromarray(input_image.transpose(1, 0, 2)))
        
        x = torch.unsqueeze(x, 0)
        y_pred = unet(x.to(device))
        y_pred_np = torch.squeeze(y_pred).detach().cpu().numpy()
        #!
        print(y_pred_np.shape)
        break # temp
        