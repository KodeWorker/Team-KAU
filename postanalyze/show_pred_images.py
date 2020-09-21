import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    
    raw_image_dir = r"D:\Datasets\Brain Tumor Segmentation Challenge\test\image"
    
    prediction_dir = r"./prediction"
    predictions = ["2-stage-224-2d-unet32", "2-stage-256-2d-unet64", "128-3d-unet16", "128-3d-unet16-pretrained"]
    
    fig_dir = "./fig/20200921"
    
    images = glob.glob(os.path.join(raw_image_dir, "*.nii.gz"))
    
    for image in tqdm(images):
        name = os.path.basename(image).replace(".nii.gz", "")
        
        if not os.path.exists(os.path.join(fig_dir, name)):
            os.makedirs(os.path.join(fig_dir, name))
        
        true_image = nib.load(image)
        x_true = true_image.get_fdata()
        
        pred_dict = {prediction: os.path.join(prediction_dir, prediction, "{}.nii.gz".format(name)) for prediction in predictions}
        
        y_pred_dict = {}
        for key, value in pred_dict.items():
            pred_mask = nib.load(value)
            y_pred_dict[key] = pred_mask.get_fdata()
        
        fig_row, fig_col = 2, 3
        for n_slice in range(x_true.shape[-1]):
        
            xn = x_true[...,n_slice].transpose(1,0)
           
            if np.max(xn) - np.min(xn) != 0:
                xn = (xn - np.min(xn)) / (np.max(xn) - np.min(xn)) * 255
            xn = xn.astype(np.uint8)
            
            fig, ax = plt.subplots(fig_row, fig_col, figsize=(8,8))
    
            ax[0][0].set_title("origin")
            ax[0][0].imshow(xn, alpha=1.0)
            
            ###
            y_pred_dict_ = {key: value[...,n_slice].transpose(1,0) * 255 for key, value in y_pred_dict.items()}
            
            for key, value in y_pred_dict_.items():
                ind = predictions.index(key)
                
                j = (ind % (fig_col -1)) + 1 # first col for origin
                i = ind // (fig_col -1)
                
                ax[i][j].set_title(key)
                ax[i][j].imshow(value, alpha=0.8, cmap=plt.cm.get_cmap("Reds"))
                ax[i][j].imshow(xn, alpha=0.3)
                for i in range(1, fig_row):
                    ax[i][0].axis('off')
            ###
            
            filename = os.path.join(fig_dir, name, "{:04d}.jpg".format(n_slice))
            fig.savefig(filename)
            plt.close(fig)