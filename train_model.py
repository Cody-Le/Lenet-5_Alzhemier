import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf




def PILtoNumpy(pilObject):
    width, height = pilObject.size
    pixels = list(pilObject.getdata())
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

    return np.array(pixels)


def to_directory():
    dir_data = ""


if __name__ == "__main__":
    batch_size = 32
    image_height, image_width = 176, 208
    df = tf.keras.utils.image_dataset_from_directory("./Alzheimer_s Dataset/train",
                                                     image_size=(image_width, image_height),
                                                     batch_size=batch_size,
                                                     color_mode="grayscale"
                                                     )
    print(df)
    classNames = df.class_names
    plt.figure(figsize=(10, 10))

    for images, labels in df.take(8):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            print(images[i].numpy().shape)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(classNames[labels[i]])
            plt.axis("off")

    

    if (False):
        plt.show()



    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = (208, 176,1)))
    model.add( tf.keras.layers.Conv2D(6, (5,5), activation="relu"))
    model.add( tf.keras.layers.AvgPool2D(2,2))
    model.add(tf.keras.layers.Conv2D(16, (5,5), activation="relu"))
    model.add(tf.keras.layers.AvgPool2D(2, 2))
    model.add(tf.keras.layers.Conv2D(120, (5,5),activation="relu"))
    model.add(tf.keras.layers.AvgPool2D(4,4))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(84, activation="relu"))
    model.add(tf.keras.layers.Dense(4, activation="relu"))
    model.add(tf.keras.layers.Softmax())

    model.compile(optimizer="sgd", loss = "mse")
    model.summary()


    checkpoint_path = "../savedModel/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose = 1
                                                     )

    model.fit(df, epochs=10, callbacks=[cp_callback])








