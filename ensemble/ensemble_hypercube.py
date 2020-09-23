import os
import glob
from tqdm import tqdm
import nibabel as nib
import numpy as np
import cv2
import random

def crop_cube(cube, index, cube_size):

    output = np.zeros((cube_size, cube_size, cube_size))
    
    x_start, x_end = index[0] - cube_size//2, index[0] + cube_size//2 + 1
    y_start, y_end = index[1] - cube_size//2, index[1] + cube_size//2 + 1
    z_start, z_end = index[2] - cube_size//2, index[2] + cube_size//2 + 1
    
    output = cube[x_start: x_end, y_start: y_end, z_start: z_end]
    return output

if __name__ == "__main__":
    
    feature_dir = "./predictions(validation)"
    label_dir = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\validate_label\label"
    """
    score = {
             "2-stage-224-2d-unet32": 0.5475687 * 0.2,
             "2-stage-256-2d-unet64": 0.5316030 * 0.2,
             "2-stage-256-2d-rc-unet64": 0.4642530 * 0.2,
             "2-stage-256-2d-n4rc-unet64": 0.4782162 * 0.2,
             "2-stage-224-2d-n4-unet32": 0.4700288 * 0.2,
             
             "128-3d-unet16": 0.5915776 * 0.5,
             "128-3d-unet16-pretrained": 0.6051299 * 0.5
            }
    """
    cube_size = 5
    prob_lb = 1e-5
    output_dir = "./rf_features"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_list = os.listdir(feature_dir)
    
    filenames = glob.glob(os.path.join(feature_dir, model_list[0], "*.nii.gz"))
    filenames = [os.path.basename(filename) for filename in filenames]
    
    total_possitive = []
    total_negative = []
    
    for i, filename in tqdm(enumerate(filenames)):
        
        label_path = os.path.join(label_dir, filename)
        label_cube = nib.load(label_path).get_fdata()
        
        feature_paths = {}
        for model in model_list:
            feature_paths[model] = os.path.join(feature_dir, model, filename)
        
        features = {key: nib.load(value).get_fdata() for key, value in feature_paths.items()}
        feature_cubes = list(features.values())
        
        # Gethering possitive cases
        #zero_count = 0
        positive_features = []
        indices = np.argwhere(label_cube == 1)
        for index in indices:
            cubes = [np.expand_dims(crop_cube(cube, index, cube_size), axis=0) for cube in feature_cubes]
            hypercube = np.concatenate(cubes, axis=0)
            
            if np.sum(hypercube>=prob_lb) != 0: # reject hybercube which cannot be predict
                #zero_count += 1
                positive_features += [hypercube.flatten()]
                
        # Gethering negative cases
        num_possitive_cases = len(possitive_features)
        negative_features = []
        indices = np.argwhere(label_cube == 0)
        random.shuffle(indices)
        for index in indices[:num_possitive_cases]:
            cubes = [np.expand_dims(crop_cube(cube, index, cube_size), axis=0) for cube in feature_cubes]
            hypercube = np.concatenate(cubes, axis=0)
            
            negative_features += [hypercube.flatten()]
        
        total_positive += positive_features
        total_negative += negative_features
        
    total_positive = np.array(total_positive)
    total_negative = np.array(total_negative)
    
    np.save(os.path.join(output_dir, "positive.npy"), total_possitive)
    np.save(os.path.join(output_dir, "negative.npy"), total_negative)
    