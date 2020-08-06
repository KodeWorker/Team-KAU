import os
import numpy as np
import tensorflow as tf
from tf2_unet import build_dataset
from unet import utils
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    data_dir = "../data_test_run"
    unet_save_path = "./unet_test_run"
    fig_dir = "./fig/unet_test_run"
    shape = (224, 224)
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    validation_dataset = build_dataset(os.path.join(data_dir, "validation"), ext="tif", shape=shape)
    # train_dataset = build_dataset(os.path.join(data_dir, "train"), ext="tif", shape=shape)
    
    model = tf.saved_model.load(unet_save_path)
    output = model(tf.constant(np.random.rand(1, 224, 224, 1).astype(np.float32)))
    
    dataset = validation_dataset.map(utils.crop_image_and_label_to_shape(output.shape[1:]))
    
    for (image, label), (cropped_image, cropped_label) in zip(validation_dataset, dataset):
    # for (image, label), (cropped_image, cropped_label) in zip(train_dataset, dataset):
        prediction = model(tf.constant(np.expand_dims(image, axis=0).astype(np.float32)))
        
        plt.imshow(image)
        plt.show()
        break