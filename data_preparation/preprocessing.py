import cv2
import numpy as np

def normalization(epi_image_data, epi_label_data):
    image_data = np.zeros(epi_image_data.shape, dtype=np.uint8)
    label_data = epi_label_data.copy()
    
    for n_slice in range(epi_image_data.shape[-1]):
        
        image = epi_image_data[...,n_slice]
        p75 = np.percentile(image, 75)
        mean, std = np.mean(image), np.std(image)
        #mean, std = np.mean(image[image > p75]), np.std(image[image > p75])
        
        image[image < p75] = 0
        
        x_min, x_max = mean, mean + 5 * std
        image = (image - x_min) / (x_max - x_min)
        image = np.clip(image, 0, 1)
        
        image_data[...,n_slice] = (image * 255).astype(np.uint8)
    
    label_data = (label_data * 255).astype(np.uint8)
    return image_data, label_data

def stack_channels(epi_image_data, epi_label_data):
    image_data = epi_image_data.copy()
    label_data = epi_label_data.copy()
    
    image_data = np.expand_dims(image_data, axis=-2)
    repeats = (1,1,3,1)
    image_data = np.tile(image_data, repeats)
    
    for n_slice in range(1, image_data.shape[-1]-1):
        slice1 = epi_image_data[...,n_slice-1]
        slice2 = epi_image_data[...,n_slice]
        slice3 = epi_image_data[...,n_slice+1]
        image_data[...,n_slice] = np.concatenate((np.expand_dims(slice1, axis=-1), 
                                                  np.expand_dims(slice2, axis=-1), 
                                                  np.expand_dims(slice3, axis=-1)), axis=-1)
    return image_data, label_data

def pad_and_resize(epi_image_data, epi_label_data, image_size):
    image_data = np.zeros((image_size, image_size, 3, epi_image_data.shape[-1]), dtype=np.uint8)
    label_data = np.zeros((image_size, image_size, epi_label_data.shape[-1]), dtype=np.uint8)
    
    for n_slice in range(epi_image_data.shape[-1]):
        image = epi_image_data[..., n_slice]
        label = epi_label_data[..., n_slice]
        
        image_square_size = max(image.shape[0], image.shape[1])
        label_square_size = max(label.shape[0], label.shape[1])        
        
        if image.shape[0] > image.shape[1]:
            image_x_init = 0
            image_y_init = int(0.5 * image_square_size - 0.5 * image.shape[1])
        else:
            image_x_init = int(0.5 * image_square_size - 0.5 * image.shape[0])
            image_y_init = 0
        if label.shape[0] > label.shape[1]:
            label_x_init = 0
            label_y_init = int(0.5 * label_square_size - 0.5 * label.shape[1])
        else:
            label_x_init = int(0.5 * label_square_size - 0.5 * label.shape[0])
            label_y_init = 0
        
        pad_image = np.zeros((image_square_size, image_square_size, 3), dtype=np.uint8)
        pad_label = np.zeros((label_square_size, label_square_size), dtype=np.uint8)
        
        pad_image[image_x_init:image_x_init+image.shape[0], image_y_init:image_y_init+image.shape[1], :] = image
        pad_label[label_x_init:label_x_init+label.shape[0], label_y_init:label_y_init+label.shape[1]] = label
        
        resize_image = cv2.resize(pad_image, (image_size, image_size), cv2.INTER_CUBIC)
        resize_label = cv2.resize(pad_label, (image_size, image_size), cv2.INTER_CUBIC)
        image_data[..., n_slice] = resize_image
        label_data[..., n_slice] = resize_label
    
    return image_data, label_data

def resample(epi_image_data, epi_label_data, pixdims):
    image_new_h = int(pixdims[0][1] * epi_image_data.shape[0])
    image_new_w = int(pixdims[0][2] * epi_image_data.shape[1])
    
    label_new_h = int(pixdims[1][1] * epi_label_data.shape[0])
    label_new_w = int(pixdims[1][2] * epi_label_data.shape[1])
    
    image_data = np.zeros((image_new_h, image_new_w, epi_image_data.shape[-1]), dtype=np.float)
    label_data = np.zeros((label_new_h, label_new_w, epi_image_data.shape[-1]), dtype=np.float)
    
    for n_slice in range(epi_image_data.shape[-1]):
        image = epi_image_data[..., n_slice]
        label = epi_label_data[..., n_slice]
        
        resize_image = cv2.resize(image, (image_new_h, image_new_w), cv2.INTER_CUBIC)
        resize_label = cv2.resize(label, (label_new_h, label_new_w), cv2.INTER_CUBIC)
        
        image_data[...,n_slice] = resize_image
        label_data[...,n_slice] = resize_label
    
    return image_data, label_data