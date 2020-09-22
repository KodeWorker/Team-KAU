import os
import glob
from tqdm import tqdm
import nibabel as nib
import numpy as np
import cv2

if __name__ == "__main__":
    
    image_size = 224
    feature_dir = "./predictions(validation)"
    label_dir = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\validate_label\label"
    output_dir = "../data/ensemble/train"
    
    if not os.path.exists(os.path.join(output_dir, "images")):
        os.makedirs(os.path.join(output_dir, "images"))
    if not os.path.exists(os.path.join(output_dir, "labels")):
        os.makedirs(os.path.join(output_dir, "labels"))
    
    model_list = os.listdir(feature_dir)
    
    filenames = glob.glob(os.path.join(feature_dir, model_list[0], "*.nii.gz"))
    filenames = [os.path.basename(filename) for filename in filenames]
    
    for filename in tqdm(filenames):
        
        label_path = os.path.join(label_dir, filename)
        label_cube = nib.load(label_path).get_fdata()
        
        feature_paths = {}
        for model in model_list:
            feature_paths[model] = os.path.join(feature_dir, model, filename)
        
        features = {key: nib.load(value).get_fdata() for key, value in feature_paths.items()}
        
        for n_slice in range(list(features.values())[0].shape[-1]):
            
            feature_list = []
            for model, cube in features.items():
                feature_list.append(np.expand_dims(cv2.resize(cube[..., n_slice], (image_size, image_size), cv2.INTER_CUBIC), axis=0))
            
            x = np.concatenate(feature_list, axis = 0)
            y = np.expand_dims(cv2.resize(label_cube[..., n_slice], (image_size, image_size), cv2.INTER_CUBIC), axis=0)
            
            output_filename = "{}_{}.npy".format(filename.replace(".nii.gz", ""), n_slice)            
            x_path = os.path.join(output_dir, "images", output_filename)
            y_path = os.path.join(output_dir, "labels", output_filename)
            np.save(x_path, x)
            np.save(y_path, y)
        #break        