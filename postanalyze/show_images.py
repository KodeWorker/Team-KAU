import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #names = ["367NGTCG.nii.gz", "3ZU763QD.nii.gz", "ADYQS7NB.nii.gz", "E7AD2EZH.nii.gz", "HQFRTSMF.nii.gz", "LJX7N4UD.nii.gz"]
    names = ["633TNGNB.nii.gz", "A4GY5J5E.nii.gz", "IYO2C7Z7.nii.gz", "JMAI55EV.nii.gz"]
    raw_validation_image_folder = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\validate_image\image"
    raw_validation_label_folder = r"D:\Datasets\Brain Tumor Segmentation Challenge\data\validation\validate_label\label"
    prediction_folder = r"./prediction/pytorch_unet(47)"
    fig_dir = "./fig"
    
    prediction_labels = glob.glob(os.path.join(prediction_folder, "*.nii.gz"))
    validation_labels = glob.glob(os.path.join(raw_validation_label_folder, "*.nii.gz"))
    validation_images = glob.glob(os.path.join(raw_validation_image_folder, "*.nii.gz"))
    
    for pred, true, image in zip(prediction_labels, validation_labels, validation_images):
        if os.path.basename(pred) in names:
            name = os.path.basename(pred).replace(".nii.gz", "")
            
            if not os.path.exists(os.path.join(fig_dir, name)):
                os.makedirs(os.path.join(fig_dir, name))
            
            pred_mask = nib.load(pred)
            true_mask = nib.load(true)
            true_image = nib.load(image)
            
            y_pred = pred_mask.get_fdata()
            y_true = true_mask.get_fdata()
            x_true = true_image.get_fdata()
            
            for n_slice in range(x_true.shape[-1]):
                xn = x_true[...,n_slice].transpose(1,0) 
                yn = y_true[...,n_slice].transpose(1,0) * 255
                yn_pred = y_pred[...,n_slice].transpose(1,0) * 255
                
                if np.max(xn) - np.min(xn) != 0:
                    xn = (xn - np.min(xn)) / (np.max(xn) - np.min(xn)) * 255
                xn = xn.astype(np.uint8)
                
                fig, ax = plt.subplots(1,2, figsize=(8,8))
        
                ax[0].imshow(yn, alpha=0.8, cmap=plt.cm.get_cmap("Greens"))
                ax[0].imshow(xn, alpha=0.3)
                
                ax[1].imshow(yn_pred, alpha=0.8, cmap=plt.cm.get_cmap("Reds"))
                ax[1].imshow(xn, alpha=0.3)
                
                filename = os.path.join(fig_dir, name, "{:04d}.jpg".format(n_slice))
                fig.savefig(filename)
                plt.close(fig)
                
            #break #!