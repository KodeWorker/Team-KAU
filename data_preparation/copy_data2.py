import os
import glob
from shutil import copyfile
from tqdm import tqdm
import cv2
import numpy as np

if __name__ == "__main__":

    source = "../data/demo/train"
    target = "../data/temp2/train"
    n_images = 5000
    
    if not os.path.exists(target):
        os.makedirs(target)
    
    if not os.path.exists(os.path.join(target, "imgs")):
        os.makedirs(os.path.join(target, "imgs"))
    
    if not os.path.exists(os.path.join(target, "masks")):
        os.makedirs(os.path.join(target, "masks"))
    
    images = glob.glob(source + "/imgs/*.png")
    
    count = 0
    for image_path in images:
        filename = os.path.basename(image_path)
        mask_path = image_path.replace("imgs", "masks")
        
        mask = cv2.imread(mask_path)
        if np.sum(mask!=0) != 0:
            copyfile(image_path, os.path.join(target, "imgs", filename))
            copyfile(mask_path, os.path.join(target, "masks", filename))
            count += 1
        
        if count == n_images:
            break