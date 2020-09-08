import os
import glob
from aidea_btsc import dice_score

if __name__ == "__main__":
    raw_validation_label_folder = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\validate_label\label"
    prediction_folder = r"./prediction/pytorch_two_stage(58)_efn+_unet64"
    
    prediction_labels = glob.glob(os.path.join(prediction_folder, "*.nii.gz"))
    validation_labels = glob.glob(os.path.join(raw_validation_label_folder, "*.nii.gz"))
    
    for pred, true in zip(prediction_labels, validation_labels):
        
        name = os.path.basename(pred)
        score = dice_score(pred, true)
        
        print("{}: {:.8f}".format(name, score))