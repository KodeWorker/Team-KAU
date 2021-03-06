import glob
import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
    
if __name__ == "__main__":

    score = {"group1": {"2-stage-224-2d-unet32":0.5817068,
                        "2-stage-256-2d-unet64":0.5896492,
                        "2-stage-256-n4rc-unet64": 0.53648260},             
             "group2": {"64-3d-unet32":0.4385284,
                        "64-3d-unet16":0.4525714,
                        "128-3d-unet16":0.5892320,
                        "128-3d-unet16-pretrained":0.595441}}
             
    predictions_dir = "./predictions"
    results_dir = "./results/voting"
    threshold = 0.5
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    filenames = [os.path.basename(filename) for filename in glob.glob(os.path.join(predictions_dir, list(score["group1"].keys())[0], "*.nii.gz"))]
    for filename in tqdm(filenames):
        
        pred_within_group = {}
        
        for group in score.keys():
        
            collection = []
            
            for model in score[group]:
                weight = score[group][model] / sum(score[group].values())
                prediction_file_path = os.path.join(predictions_dir, model, filename)
                
                pred_image = nib.load(prediction_file_path)
                y_pred = pred_image.get_fdata()
                
                collection.append(y_pred*weight)
        
            total_sum = np.sum(np.array(collection), axis=0)
            
            total_sum[total_sum<threshold] = 0
            total_sum[total_sum>=threshold] = 1
            pred_within_group[group] = total_sum
        
        x = np.logical_or(pred_within_group["group1"], pred_within_group["group2"])
        x = x.astype(np.float)
        
        #x = np.logical_and(pred_within_group["group1"], pred_within_group["group2"])
        #x = x.astype(np.float)
        
        output_file_path = os.path.join(results_dir, filename)
        img = nib.Nifti1Image(x, affine=None)
        img.to_filename(output_file_path)
        