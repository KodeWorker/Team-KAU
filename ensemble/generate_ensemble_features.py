import os
#from pytorch_unet_prediction import predict, predict_central_focus
from pytorch_two_stage_prediction import predict
import glob
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    raw_validation_image_folder = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\train\image"
    
    prediction_folder = "./features/2-stage-unet64"
    
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)
    
    validation_images = glob.glob(os.path.join(raw_validation_image_folder, "*.nii.gz"))
    
    scores = []
    for image in tqdm(validation_images):
        out_file = os.path.join(prediction_folder, os.path.basename(image))
        predict(image, out_file)