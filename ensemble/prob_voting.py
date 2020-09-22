import glob
import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    
    predictions_dir = "./predictions(test)"
    
    score = {
             "2-stage-224-2d-unet32": 0.5475687 * 0.25,
             "2-stage-256-2d-unet64": 0.5316030 * 0.25,
             "2-stage-256-2d-rc-unet64": 0.4642530 * 0.25,
             "2-stage-256-2d-n4rc-unet64": 0.4782162 * 0.25,
             
             "128-3d-unet16": 0.5915776 * 0.5,
             "128-3d-unet16-pretrained": 0.6051299 * 0.5
            }
    
    results_dir = "./results/voting"
    threshold = 0.5
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    
    filenames = [os.path.basename(filename) for filename in glob.glob(os.path.join(predictions_dir, list(score.keys())[0], "*.nii.gz"))]
    for filename in tqdm(filenames):
        
        collection = []
        score_sum = []
        
        for model in score.keys():
        
            prediction_file_path = os.path.join(predictions_dir, model, filename)
            
            pred_image = nib.load(prediction_file_path)
            y_pred = pred_image.get_fdata()
            
            if (np.sum(y_pred) != 0):
                
                score_sum.append(score[model])
                collection.append(y_pred*score[model])
        
        collection = [c/np.sum(score_sum) for c in collection]
        
        total_sum = np.sum(np.array(collection), axis=0)
        
        total_sum[total_sum<threshold] = 0
        total_sum[total_sum>=threshold] = 1
        
        output_file_path = os.path.join(results_dir, filename)
        
        img = nib.Nifti1Image(total_sum, affine=None)
        img.to_filename(output_file_path)