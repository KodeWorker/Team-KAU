import os
from pytorch_unet_prediction import predict
from aidea_btsc import dice_score
import glob
import numpy as np

if __name__ == "__main__":

    raw_validation_image_folder = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\validate_image\image"
    raw_validation_label_folder = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\validate_label\label"
    
    prediction_folder = "./prediction/pytorch_unet"
    
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)
    
    validation_images = glob.glob(os.path.join(raw_validation_image_folder, "*.nii.gz"))
    validation_labels = glob.glob(os.path.join(raw_validation_label_folder, "*.nii.gz"))
    
    scores = []
    for image, label in zip(validation_images, validation_labels):
        out_file = os.path.join(prediction_folder, os.path.basename(label))
        predict(image, out_file)
        scores.append(dice_score(out_file, label))
        break # temp
    
    score = np.mean(scores)
    print("score: {:.8f}".format(score))