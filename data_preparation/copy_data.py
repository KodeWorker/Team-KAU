import os
import glob
from shutil import copyfile
from tqdm import tqdm

if __name__ == "__main__":

    source = "../data/demo/validation"
    target = "../data/temp2/valid"
    n_images = 1000
    
    if not os.path.exists(target):
        os.makedirs(target)
    
    if not os.path.exists(os.path.join(target, "imgs")):
        os.makedirs(os.path.join(target, "imgs"))
    
    if not os.path.exists(os.path.join(target, "masks")):
        os.makedirs(os.path.join(target, "masks"))
    
    images = glob.glob(source + "/imgs/*.png")
    
    for image_path in tqdm(images[:n_images]):
        filename = os.path.basename(image_path)
        mask_path = image_path.replace("imgs", "masks")
        
        copyfile(image_path, os.path.join(target, "imgs", filename))
        copyfile(mask_path, os.path.join(target, "masks", filename))
        