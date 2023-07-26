import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow_datasets as tfds

from sklearn.preprocessing import LabelEncoder

# import tensorflow_addons as tfa


'''
useful links:

https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/keras_example.ipynb#scrollTo=XWqxdmS1NLKA
https://www.kaggle.com/code/kutaykutlu/resnet50-transfer-learning-cifar-10-beginner
'''


def choose_gpu(gpu_id=0):
    # List all available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Restrict TensorFlow to only use the GPU with index 1
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            print("GPU " + str(gpu_id) + " selected")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def formatted_datetime():
    # current date and time
    now = str(datetime.now())
    now = now.replace(" ", "_")
    now = now.replace(".", "_")
    now = now.replace(":", "_")

    return now


def load_resnet_model(summary=False, include_top=False, weights="imagenet", model_trainable=True):
    '''
    We are relying on Transfer learning, using a pretrained ResNet50 model, and readapt it
    in order to be used in our use case

    "include_top" refers to the final layer or set of layers of the model,
    which are typically fully connected layers used for classification, so we are
    excluding them since we have a different dataset (cifar) to classify

    with include_top=False, it doesn't know the exact dimensions of the output tensor,
    because it depends on the input shape, which can be any size. we need to specify the input shape
    when loading the ResNet50 model.
    '''

    base_model = ResNet50(include_top=include_top, weights=weights, input_shape=(224, 224, 3))
    base_model.trainable = model_trainable

    '''
    Flatten is necessary since the base model outputs a 3D tensor and the following 
    dense layer expects a 1D tensor.

    We have 10 neurons at the end (since CIFAR-10 has 10 classes). 
    The model outputs the raw values for each class, which are then transformed 
    into probabilities by the softmax function applied in the loss function.
    '''

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10))

    if (summary):
        model.summary()

    optimizer = optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    print("optimizer: ", optimizer.get_config()['name'])
    print("loss: ", loss.get_config()['name'])

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model


def save_model(model, history, model_path_filename, history_path_filename):
    now = formatted_datetime()

    model_path_filename = model_path_filename + "_" + now

    model.save(model_path_filename)

    history_complete_path = model_path_filename + '/' + history_path_filename + '.npy'

    np.save(history_complete_path, history.history)

    print("Model saved to: " + model_path_filename)
    print("Training History saved to: " + history_complete_path)


def preprocess_img(image, label):
    '''
    We need to resize the image to 224 since ResNet50 expects that as input
    '''

    image = tf.image.resize(image, [224, 224])
    return tf.cast(image, tf.float32) / 255., label


def load_cifar10():
    (ds_train, ds_test), ds_info = tfds.load('cifar10', split=['train', 'test'], shuffle_files=True, with_info=True,
                                             as_supervised=True)

    '''
    This line applies the normalize_img function to each element in the ds_train dataset.
    The map function applies a given function to each element in the dataset. num_parallel_
    calls=tf.data.AUTOTUNE tells TensorFlow to choose an optimal number of threads to run
    the function in parallel, improving loading speed.
    '''

    ds_train = ds_train.map(
        preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

    '''
    tf.data.Dataset.cache As you fit the dataset in memory, cache it before shuffling for a better performance.
    Note: Random transformations should be applied after caching.

    tf.data.Dataset.shuffle: For true randomness, set the shuffle buffer to the full dataset size.
    Note: For large datasets that can't fit in memory, use buffer_size=1000 if your system allows it.

    tf.data.Dataset.batch: Batch elements of the dataset after shuffling to get unique batches at each epoch.

    tf.data.Dataset.prefetch: It is good practice to end the pipeline by prefetching for performance.
    '''
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    '''
    Tesing pipeline is similar

    Caching is done after batching because batches can be the same between epochs.
    '''

    ds_test = ds_test.map(
        preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_info


def fit_resnet_and_save_model(model_path_filename, history_path_filename, summary=True, epochs=5):
    print("epochs: ", epochs)

    model = load_resnet_model(summary=summary)

    ds_train, ds_test, ds_info = load_cifar10()

    # Create EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        min_delta=0.001,  # Minimum change to qualify as an improvement
        patience=10,  # Number of epochs with no improvement to stop training
        verbose=1,  # Report training progress
        restore_best_weights=True
        # Whether to restore model weights from the epoch with the best value of the monitored quantity.
    )

    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
        callbacks=[early_stopping]
    )

    save_model(model, history, model_path_filename, history_path_filename)


if __name__ == "__main__":
    print('Tensorflow ', tf.__version__)

    # Where to save/load the fitted model and its history file
    model_path_filename = "tf_saved_models/resnet50_cifar10"
    history_path_filename = "classification_history_model"

    epochs = 100

    gpu_id = 0

    choose_gpu(gpu_id=gpu_id)

    fit_resnet_and_save_model(model_path_filename, history_path_filename, summary=True, epochs=epochs)