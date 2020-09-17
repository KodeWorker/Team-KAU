import os
#from pytorch_unet_prediction import predict, predict_central_focus
from aidea_btsc import dice_score
import glob
import numpy as np

if __name__ == "__main__":

    #raw_validation_image_folder = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\validate_image\image"
    raw_validation_label_folder = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\validate_label\label"
    
    prediction_folder = r"D:\code\AIdea\inference\prediction\2-stage-unet64-n4rc(0.53648260)"
    
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)
    
    #validation_images = glob.glob(os.path.join(raw_validation_image_folder, "*.nii.gz"))
    validation_labels = glob.glob(os.path.join(raw_validation_label_folder, "*.nii.gz"))
    
    scores = []
    for label in validation_labels:
        out_file = os.path.join(prediction_folder, os.path.basename(label))
        #predict_central_focus(image, out_file)
        #predict(image, out_file)
        score_ = dice_score(out_file, label)
        scores.append(score_)
    
        print(score_)
    score = np.mean(scores)
    print("score: {:.8f}".format(score))