import glob
import os
from tqdm import tqdm
import numpy as np
import nibabel as nib

if __name__ == "__main__":
    unet_output_dir = r"D:\code\AIdea\ensemble\ensemble_unet\results\unet64"
    voting_output_dir = r"D:\code\AIdea\ensemble\ensemble_unet\results\prob_voting"
    output_dir = "./results/union"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    unet_files = glob.glob(os.path.join(unet_output_dir, "*.nii.gz"))
    voting_files = glob.glob(os.path.join(voting_output_dir, "*.nii.gz"))
    
    for unet_file, voting_file in tqdm(zip(unet_files, voting_files)):
        
        filename = os.path.basename(unet_file)
        
        unet_cube = nib.load(unet_file).get_fdata()
        voting_cube = nib.load(voting_file).get_fdata()
        
        union = np.logical_or(unet_cube, voting_cube).astype(np.float)
        
        print("unet: {}, voting: {}, union: {}".format(np.count_nonzero(unet_cube), np.count_nonzero(voting_cube), np.count_nonzero(union)))
        
        out_file = os.path.join(output_dir, filename)
        img = nib.Nifti1Image(union, affine=None)
        img.to_filename(out_file)