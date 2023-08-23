import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf

import tensorflow_datasets as tfds

def choose_gpu(gpu_id=0):
    # List all available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Restrict TensorFlow to only use the GPU with index=gpu_id
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

def save_model(model, history, model_path_filename, history_path_filename, save_weights=False):
    now = formatted_datetime()

    model_path_filename = model_path_filename + "_" + now

    if (save_weights):
        # saving nested models can cause problems, i.e., crashing at save phase
        # save model's weights instead. downside of this method is that you will
        # have to define the model architecture again before loading the weights
        model.save_weights(model_path_filename)
    else:
        model.save(model_path_filename)

    history_complete_path = model_path_filename + '/' + history_path_filename + '.npy'

    np.save(history_complete_path, history.history)

    print("Model saved to: " + model_path_filename)
    print("Training History saved to: " + history_complete_path)

def load_model(model_path_filename, history_path_filename):
    if os.path.exists(model_path_filename):
        model = tf.keras.models.load_model(model_path_filename)

        history_complete_path = model_path_filename + '/' + history_path_filename + '.npy'

        history = np.load(history_complete_path, allow_pickle='TRUE').item()

        print("Loading model: " + model_path_filename + "/\nLoading history: " + history_complete_path)

        return model, history
    else:
        print("Model not found!")

        return None, None