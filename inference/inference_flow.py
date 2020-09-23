import os
#from pytorch_unet_prediction import predict, predict_central_focus
from pytorch_two_stage_feature_generation import predict
#from pytorch_two_stage_prediction import predict
#from pytorch_two_stage_ncrops import predict
from aidea_btsc import dice_score
import glob
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    #raw_validation_image_folder = r"D:\Datasets\Brain Tumor Segmentation Challenge\test\n4"
    raw_validation_image_folder = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\n4"
    
    prob_folder = "./predictions(validation)/2-stage-224-2d-n4-unet32"
    label_folder = "./predictions/2-stage-224-2d-n4-unet32"
    
    if not os.path.exists(prob_folder):
        os.makedirs(prob_folder)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    
    validation_images = glob.glob(os.path.join(raw_validation_image_folder, "*.nii.gz"))
    #validation_labels = glob.glob(os.path.join(raw_validation_label_folder, "*.nii.gz"))
    
    scores = []
    #for image, label in zip(validation_images, validation_labels):
    for image in tqdm(validation_images):
        out_prob = os.path.join(prob_folder, os.path.basename(image))
        out_label = os.path.join(label_folder, os.path.basename(image))
        #predict_central_focus(image, out_file)
        predict(image, out_prob, out_label)
        #score_ = dice_score(out_file, label)
        #scores.append(score_)
    
        #print(score_)
    #score = np.mean(scores)
    #print("score: {:.8f}".format(score))