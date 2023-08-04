import tensorflow as tf

import mobilenetv2_imagenette
from utils.util import *

# Replace with your directory path
directory = "ds/"

# Define the batch size and image size
batch_size = 32
img_height = 224
img_width = 224

# Generate the dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# # Generate the dataset
# val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     directory,
#     image_size=(img_height, img_width),
#     batch_size=batch_size)

# Apply the preprocessing function to the dataset
train_dataset = train_dataset.map(mobilenetv2_imagenette.preprocess_img)
val_dataset = val_dataset.map(mobilenetv2_imagenette.preprocess_img)

model = tf.keras.applications.MobileNetV2(include_top=True, weights="imagenet")
model.trainable = False

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy',tf.keras.metrics.SparseCategoricalAccuracy()])

model.evaluate(val_dataset)