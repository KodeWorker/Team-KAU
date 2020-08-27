import os
import glob
import cv2
import numpy as np

if __name__ == "__main__":
    
    nobrain_folder = r"D:\code\AIdea\data\classifier\nobrain"
    label_folder = r"D:\code\AIdea\data\demo\train\masks"
    
    nobrain_images = glob.glob(nobrain_folder + "/*.png")
    
    nobrain_filename = [os.path.basename(image) for image in nobrain_images]
    nobrain_labels = [label for label in glob.glob(label_folder + "/*.png") if os.path.basename(label) in nobrain_filename]
    
    total_count = 0
    
    for image, label in zip(nobrain_images, nobrain_labels):
        name = os.path.basename(image)
        
        mask = cv2.imread(label)
        count = np.count_nonzero(mask)
        total_count += count
        
        print("{}: {:d}".format(name, count))
    print("total count: {:d}".format(total_count))