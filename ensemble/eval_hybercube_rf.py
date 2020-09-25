from sklearn.ensemble import RandomForestRegressor
from joblib import load
import numpy as np
import os
from tqdm import tqdm
import glob
import nibabel as nib
import cv2

def crop_cube(cube, index, cube_size):

    output = np.zeros((cube_size, cube_size, cube_size))
    
    x_start, x_end = index[0] - cube_size//2, index[0] + cube_size//2 + 1
    y_start, y_end = index[1] - cube_size//2, index[1] + cube_size//2 + 1
    z_start, z_end = index[2] - cube_size//2, index[2] + cube_size//2 + 1
    
    output = cube[x_start: x_end, y_start: y_end, z_start: z_end]
    return output

if __name__ == "__main__":
	
    feature_dir = "./predictions(test)"
    cube_size = 5
    prob_lb = 1e-5
    output_dir = "./results/hypercube"
    model_path = "rf.model"
    threshold = 0.5
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    model_list = os.listdir(feature_dir)
    
    filenames = glob.glob(os.path.join(feature_dir, model_list[0], "*.nii.gz"))
    filenames = [os.path.basename(filename) for filename in filenames]
    
    rf = load(model_path)
    
    for i, filename in tqdm(enumerate(filenames)):
        
        feature_paths = {}
        for model in model_list:
            feature_paths[model] = os.path.join(feature_dir, model, filename)
        
        features = {key: nib.load(value).get_fdata() for key, value in feature_paths.items()}
        feature_cubes = list(features.values())
        
        target_cubes = [np.expand_dims(cube, axis=0) for cube in feature_cubes]
        target_cube = np.max(target_cubes, axis=0)
        target_cube[target_cube < prob_lb] = 0
        
        x, y, z = target_cube.shape[1], target_cube.shape[2], target_cube.shape[3]
        label_cube = np.zeros((x, y, z))
        
        indices = np.argwhere(target_cube[0] != 0)
        
        features = []
        sel_indices = []
        for index in indices:
        
            if (index[0] - cube_size//2) > 0 and (index[0] + cube_size//2 + 1) < x and\
               (index[1] - cube_size//2) > 0 and (index[1] + cube_size//2 + 1) < y and\
               (index[2] - cube_size//2) > 0 and (index[2] + cube_size//2 + 1) < z :
               
                cubes = [np.expand_dims(crop_cube(cube, index, cube_size), axis=0) for cube in feature_cubes]
                hypercube = np.concatenate(cubes, axis=0)
                
                features += [hypercube.flatten()]
                sel_indices += [index]
        
        X = np.array(features)
        y = rf.predict(X)
        
        true_indices = np.array(sel_indices)[y >= threshold]
        for true_index in true_indices:
            label_cube[true_index] = 1
        
        for n_slice in range(label_cube.shape[-1]):
            cv2.imwrite(os.path.join(output_dir, "{}_{}.jpg".format(filename.replace(".nii.gz", ""), n_slice)), label_cube[..., n_slice]*255)
        
        output_file_path = os.path.join(output_dir, filename)
        img = nib.Nifti1Image(label_cube, affine=None)
        img.to_filename(output_file_path)
        
        break
    #
    