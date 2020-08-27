import os
import glob
from shutil import copyfile
from tqdm import tqdm
import cv2
import numpy as np
from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser()
    # model settings
    parser.add_argument("-s", "--source", help="", required=True, type=str)
    parser.add_argument("-t", "--target", help="", required=True, type=str)
    return parser
    
if __name__ == "__main__":
    args = build_argparser().parse_args()
    source = args.source
    target = args.target
    
    if not os.path.exists(target):
        os.makedirs(target)
    
    if not os.path.exists(os.path.join(target, "imgs")):
        os.makedirs(os.path.join(target, "imgs"))
    
    if not os.path.exists(os.path.join(target, "masks")):
        os.makedirs(os.path.join(target, "masks"))
    
    images = glob.glob(source + "/imgs/*.png")
    
    count = 0
    for image_path in tqdm(images):
        filename = os.path.basename(image_path)
        mask_path = image_path.replace("imgs", "masks")
        
        mask = cv2.imread(mask_path)
        if np.sum(mask!=0) != 0:
            copyfile(image_path, os.path.join(target, "imgs", filename))
            copyfile(mask_path, os.path.join(target, "masks", filename))