import matplotlib.pyplot as plt
import numpy as np
import unet
from unet import utils
from unet.datasets import circles

if __name__ == "__main__":
    
    train_dataset, validation_dataset = circles.load_data(100, nx=200, ny=200, splits=(0.7, 0.3))

    unet_model = unet.build_model(channels=circles.channels,
                                  num_classes=circles.classes,
                                  layer_depth=3,
                                  filters_root=16)
    unet.finalize_model(unet_model)
    
    trainer = unet.Trainer(checkpoint_callback=False)
    trainer.fit(unet_model,
                train_dataset,
                validation_dataset,
                epochs=5,
                batch_size=1)
                
    prediction = unet_model.predict(validation_dataset.batch(batch_size=3))
    
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8,8))
    dataset = validation_dataset.map(utils.crop_image_and_label_to_shape(prediction.shape[1:]))

    for i, (image, label) in enumerate(dataset.take(3)):
        ax[i][0].matshow(image[..., -1])
        ax[i][1].matshow(np.argmax(label, axis=-1), cmap=plt.cm.gray)
        ax[i][2].matshow(np.argmax(prediction[i,...], axis=-1), cmap=plt.cm.gray)
    plt.tight_layout()
    plt.show()