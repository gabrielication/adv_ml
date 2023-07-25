import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

import tensorflow

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split

def is_gpu_supported():
    gpu_list = tf.config.list_physical_devices('GPU')

    if (len(gpu_list) == 0):
        print("GPU IS NOT SUPPORTED/ACTIVE/DETECTED!")

        return False
    else:
        print("GPU SUPPORTED: ", gpu_list)

        return True

def formatted_datetime():
    # current date and time
    now = str(datetime.now())
    now = now.replace(" ", "_")
    now = now.replace(".", "_")
    now = now.replace(":", "_")

    return now

def load_model(summary=False, include_top=True, weights="imagenet", model_trainable=False):
    model = ResNet50(include_top=include_top, weights=weights)
    model.trainable = model_trainable

    if(summary):
        model.summary()

    return model

def load_images(img_folder_path, img_filename="val_", img_type=".JPEG"):

    # ResNet50 was trained on images of size (224, 224) from the ImageNet dataset

    images = []

    for i in range(5):
        i_str = str(i)

        complete_path = img_folder_path+img_filename+i_str+img_type

        # target_size is going to resize the image to the desired resolution
        img = load_img(complete_path, target_size=(224,224))

        img_array = img_to_array(img)

        images.append(img_array)

    images = np.array(images)

    images_preprocessed = preprocess_input(images)

    return images, images_preprocessed

def generate_labels(val_folder_path, val_filename):

    dict = {}

    with open(val_folder_path+val_filename, 'r') as file:
        for line in file:
            splitted = line.split()
            filename = splitted[0]
            label = splitted[1]

            dict[filename] = label

    return dict

if __name__ == "__main__":

    print('Tensorflow ', tf.__version__)

    is_gpu_supported()

    model = load_model()

    img_folder_path = "images/tiny-imagenet-200/val/images/"

    images, images_preprocessed = load_images(img_folder_path)

    val_folder_path = "images/tiny-imagenet-200/val/"
    val_filename = "val_annotations.txt"

    label_dict = generate_labels(val_folder_path, val_filename)

    print()



