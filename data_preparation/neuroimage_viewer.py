import os
import cv2
import glob
import nibabel as nib
import numpy as np
import imutils

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
    
    fold = "train"
    data_dir = r"D:\Datasets\Brain Tumor Segmentation Challenge\data"
    if fold == "train":
        images = glob.glob(os.path.join(data_dir, fold) + "\image\*.nii.gz")
        labels = glob.glob(os.path.join(data_dir, fold) + "\label\*.nii.gz")
    elif fold == "validation":
        images = glob.glob(os.path.join(data_dir, fold) + "\*\image\*.nii.gz")
        labels = glob.glob(os.path.join(data_dir, fold) + "\*\label\*.nii.gz")
    
    n_files = 0
    
    image_path = images[n_files]
    label_path = labels[n_files]
    
    epi_image = nib.load(image_path)
    epi_label = nib.load(label_path)
    
    epi_image_data = epi_image.get_fdata()
    epi_label_data = epi_label.get_fdata()
    
    n_slices = 0
    
    epi_image_record, epi_label_record = generate_crop_images(epi_image_data, epi_label_data)
    
    while True:
        update = False
        
        color_image = cv2.cvtColor(epi_image_record[n_slices], cv2.COLOR_GRAY2BGR)
        color_mask = cv2.applyColorMap(epi_label_record[n_slices]*255, cv2.COLORMAP_JET)
        
        combined_image = cv2.addWeighted(color_image,0.8,color_mask,0.2,0)
        
        cv2.imshow("SCAN", combined_image)
        
        keyPress = cv2.waitKey(20)
        if(keyPress == ord("q")):
            break;
        
        elif(keyPress == ord("a")):
            n_slices -= 1
        elif(keyPress == ord("d")): 
            n_slices += 1
        
        elif(keyPress == ord("z")):
            n_files -= 1
            update = True
        elif(keyPress == ord("c")): 
            n_files += 1
            update = True
        
        n_files = n_files % len(images)
        n_slices = n_slices % epi_image_data.shape[-1]
        
        if update:
            n_slices = 0
        
            image_path = images[n_files]
            label_path = labels[n_files]
            
            epi_image = nib.load(image_path)
            epi_label = nib.load(label_path)
            
            epi_image_data = epi_image.get_fdata()
            epi_label_data = epi_label.get_fdata()
            
            epi_image_record, epi_label_record = generate_crop_images(epi_image_data, epi_label_data)