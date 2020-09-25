import glob
import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    prob_dir = "./results/unet64_prob_exam"
    #prob_dir = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\validate_label\label"
    threshold = 0.5
    output_dir = "./results/unet64_postprocess"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepaths = glob.glob(os.path.join(prob_dir, "*.nii.gz"))
    
    for filepath in filepaths:
        filename =os.path.basename(filepath)
        
        prob_cube = nib.load(filepath).get_fdata()
        
        percentage = np.sum(prob_cube >= threshold) / (prob_cube.shape[0]*prob_cube.shape[1]*prob_cube.shape[2]) * 100
        
        if percentage <= 0.0015:
            threshold = np.percentile(prob_cube, 99.9995)            
            percentage = np.sum(prob_cube >= threshold) / (prob_cube.shape[0]*prob_cube.shape[1]*prob_cube.shape[2]) * 100
        
        prob_cube[prob_cube >= threshold] = 1
        prob_cube[prob_cube < threshold] = 0
        
        print("{}: {:.4f} %".format(filename, percentage))
        
        output_file_path = os.path.join(output_dir, filename)
        img = nib.Nifti1Image(prob_cube, affine=None)
        img.to_filename(output_file_path)