import os
import cv2
import glob
import imutils
import numpy as np
import nibabel as nib
from tqdm import trange
import imageio
def generate_contour(image, blur_kernel_size=5, thresh=(45, 255), iterations=2):
    gray = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    #thresh = cv2.threshold(gray, thresh[0], thresh[1], cv2.THRESH_BINARY)[1]
    
    thresh = cv2.threshold(gray,thresh[0],thresh[1],cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    thresh = cv2.erode(thresh, None, iterations=iterations)
    thresh = cv2.dilate(thresh, None, iterations=iterations)
    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    contour = max(cnts, key=cv2.contourArea)
    return contour

def crop_contour(contour):        
    extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
    extRight = tuple(contour[contour[:, :, 0].argmax()][0])
    extTop = tuple(contour[contour[:, :, 1].argmin()][0])
    extBot = tuple(contour[contour[:, :, 1].argmax()][0])
    
    xmin = extLeft[0]
    xmax = extRight[0]
    ymin = extTop[1]
    ymax = extBot[1]
    
    return xmin, xmax, ymin, ymax

def generate_crop_images(epi_image_data, epi_label_data):
    # find maximum brain scan contour across slices
    epi_image_record = []
    epi_label_record = []
    epi_area_record = []
    epi_contour_record = []
    skip = False
    for n in range(epi_image_data.shape[-1]):
        image_slice = epi_image_data[..., n]    # image
        label_slice = epi_label_data[..., n]    # mask
        
        mean = np.mean(image_slice)
        std = np.std(image_slice)
        image_slice = (image_slice - mean) / std 
        
        image_slice = (image_slice.astype(np.float64) - np.min(image_slice)) / np.max(image_slice) # normalize the data to 0 - 1
        image_slice = 255 * image_slice
        
        image = np.array(image_slice.T, dtype=np.uint8)
        label = np.array(label_slice.T, dtype=np.uint8)
        
        try:
            contour = generate_contour(image.copy())
            area = cv2.contourArea(contour)
            
            epi_contour_record.append(contour)
            epi_area_record.append(area)
        except:
            skip = True
    
        epi_image_record.append(image)
        epi_label_record.append(label)
    
    if not skip:
        largest_idx = np.argmax(epi_area_record)
        xmin,xmax,ymin,ymax = crop_contour(epi_contour_record[largest_idx])
        epi_image_record = [image_record[ymin:ymax, xmin:xmax] for image_record in epi_image_record]
        epi_label_record = [label_record[ymin:ymax, xmin:xmax] for label_record in epi_label_record]
    
    return epi_image_record, epi_label_record

if __name__ == "__main__":
    
    data_dir = r"D:\Datasets\Brain Tumor Segmentation Challenge\data"
    
    for fold in ["train", "validation"]:
        
        print("Data Preparation: {}".format(fold))
        
        if fold == "train":
            images = glob.glob(os.path.join(data_dir, fold) + "\image\*.nii.gz")
            labels = glob.glob(os.path.join(data_dir, fold) + "\label\*.nii.gz")
        elif fold == "validation":
            images = glob.glob(os.path.join(data_dir, fold) + "\*\image\*.nii.gz")
            labels = glob.glob(os.path.join(data_dir, fold) + "\*\label\*.nii.gz")
        output_dir = os.path.join("../data", fold)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
                
        for n_files in trange(0, len(images)):
        
            image_path = images[n_files]
            label_path = labels[n_files]
            
            pname = os.path.basename(image_path).split(".")[0]
            
            epi_image = nib.load(image_path)
            epi_label = nib.load(label_path)
            
            epi_image_data = epi_image.get_fdata()
            epi_label_data = epi_label.get_fdata()
            
            epi_image_record, epi_label_record = generate_crop_images(epi_image_data, epi_label_data)
            
            pfolder = os.path.join(output_dir, pname)
            if not os.path.exists(pfolder):
                os.makedirs(pfolder)
            
            for n_slices in range(epi_image.shape[-1]):
                
                image_filename = "{}_{}.tif".format(pname, n_slices+1)
                mask_filename = "{}_{}_mask.tif".format(pname, n_slices+1)
                
                imageio.imwrite(os.path.join(pfolder, image_filename), epi_image_record[n_slices])
                imageio.imwrite(os.path.join(pfolder, mask_filename), epi_label_record[n_slices])