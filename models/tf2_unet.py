import os
import glob
from skimage.io import imread
import numpy as np
import cv2
import tensorflow as tf
import unet
from tqdm import tqdm

def pad_and_resize(image, label, shape):
    size = max(image.shape)
    process_image = np.zeros((size, size), dtype=np.uint8)
    process_label = np.zeros((size, size), dtype=np.uint8)
    
    if image.shape[0] == size:
        x_start = 0
        y_start = int((size - image.shape[1])/2)
    else:
        x_start = int((size - image.shape[0])/2)
        y_start = 0
    
    process_image[x_start:x_start + image.shape[0], y_start:y_start + image.shape[1]] = image
    process_label[x_start:x_start + image.shape[0], y_start:y_start + image.shape[1]] = label
    
    process_image = cv2.resize(process_image, shape, interpolation=cv2.INTER_CUBIC)
    process_label = cv2.resize(process_label, shape, interpolation=cv2.INTER_CUBIC)
    
    process_image = np.expand_dims(process_image, axis=-1)
    process_label = np.expand_dims(process_label, axis=-1)
    process_label = np.repeat(process_label, 2, axis=-1)
    
    process_label[..., -1] = 1 - process_label[..., -1]
    
    return process_image, process_label

def build_dataset(data_dir, ext, shape):
    paths = glob.glob(data_dir + "/*/*.{}".format(ext))
    image_paths = sorted([path for path in paths if "mask" not in path])
    label_paths = sorted([path for path in paths if "mask" in path])
    images, labels = list(), list()
    for image_path, label_path in tqdm(zip(image_paths, label_paths)):
        image = imread(image_path)
        label = imread(label_path)
        image, label = pad_and_resize(image, label, shape)
        images.append(np.array(image, dtype=np.float64))
        labels.append(np.array(label, dtype=np.float64))
    dataset = tf.data.Dataset.from_tensor_slices((np.array(images), np.array(labels)))
    return dataset
        
def build_train_val_dataset(data_dir, shape):
    # build train_dataset
    train_dataset = build_dataset(data_dir=os.path.join(data_dir, "train"), ext="tif", shape=shape)
    # build validation_dataset
    validation_dataset = build_dataset(os.path.join(data_dir, "validation"), ext="tif", shape=shape)
    return train_dataset, validation_dataset
    
if __name__ == "__main__":
    
    data_dir = "../data_test_run"
    unet_save_path = "./unet_test_run_2"
    shape = (224, 224)
    
    if not os.path.exists(unet_save_path):
        os.makedirs(unet_save_path)
    
    train_dataset, validation_dataset = build_train_val_dataset(data_dir, shape)

    unet_model = unet.build_model(channels=1,
                                  num_classes=2,
                                  layer_depth=3,
                                  filters_root=64)
    unet.finalize_model(unet_model)
    
    trainer = unet.Trainer(checkpoint_callback=False)
    trainer.fit(unet_model,
                train_dataset,
                validation_dataset,
                epochs=1,
                batch_size=1)
    
    tf.saved_model.save(unet_model, unet_save_path)
