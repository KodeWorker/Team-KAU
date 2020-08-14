import os
import glob
from shutil import copyfile
from tqdm import tqdm

if __name__ == "__main__":

    source = "../data/demo/train"
    target = "../data/temp/train"
    n_images = 5000
    
    if not os.path.exists(target):
        os.makedirs(target)
    
    if not os.path.exists(os.path.join(target, "imgs")):
        os.makedirs(os.path.join(target, "imgs"))
    
    if not os.path.exists(os.path.join(target, "masks")):
        os.makedirs(os.path.join(target, "masks"))
    
    images = glob.glob(source + "/imgs/*.png")[:n_images]
    
    for image_path in tqdm(images):
        filename = os.path.basename(image_path)
        mask_path = image_path.replace("imgs", "masks")
        copyfile(image_path, os.path.join(target, "imgs", filename))
        copyfile(mask_path, os.path.join(target, "masks", filename))