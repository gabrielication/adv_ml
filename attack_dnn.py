import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
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

def clean_old_models(model_path):
    print("FRESH START ENABLED. Cleaning ALL old models and their files...")

    for filename in Path(".").glob(model_path):
        try:
            shutil.rmtree(filename)

            print(str(filename) + " deleted")
        except OSError:
            print("\nError while deleting " + str(filename) + "\n")

    print("All old files deleted.\n")

def load_model(model_path_filename, history_path_filename):
    if os.path.exists(model_path_filename):
        print("Model already exists!\nIf tensorflow versions from saved one differ then a crash might happen!")
        model = tf.keras.models.load_model(model_path_filename)

        history_complete_path = model_path_filename + '/' + history_path_filename + '.npy'

        history = np.load(history_complete_path, allow_pickle='TRUE').item()

        print("Loading model: " + model_path_filename + "/\nLoading history: " + history_complete_path)

        return model, history
    else:
        print("Previous fitted model not found!")

        return None, None

def save_model(model, history, model_path_filename, history_path_filename):
    model.save(model_path_filename)

    history_complete_path = model_path_filename + '/' + history_path_filename + '.npy'

    np.save(history_complete_path, history.history)

    print("Model saved to: " + model_path_filename)
    print("Training History saved to: " + history_complete_path)

def formatted_datetime():
    # current date and time
    now = str(datetime.now())
    now = now.replace(" ", "_")
    now = now.replace(".", "_")
    now = now.replace(":", "_")

    return now

# Define the DNN model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model

# Generate adversarial examples using FGSM
def generate_adversarial_examples(model, x, y, epsilon=0.1):
    x_adv = tf.identity(x)
    with tf.GradientTape() as tape:
        tape.watch(x_adv)
        logits = model(x_adv)
        loss = tf.keras.losses.categorical_crossentropy(y, logits)
    gradients = tape.gradient(loss, x_adv)
    signed_grad = tf.sign(gradients)
    x_adv = x_adv + epsilon * signed_grad
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv

def load_normalized_and_encoded_dataset():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Display some sample images
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(x_train[i], cmap=plt.cm.binary)
    #     plt.xlabel(y_train[i])
    # plt.show()

    # Preprocess the data
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    split_index = int(0.8 * len(x_test))

    x_test_fit = x_test[split_index:]
    y_test_fit = y_test[split_index:]

    x_test_eval = x_test[:split_index]
    y_test_eval = y_test[:split_index]

    return x_train, x_test, y_train, y_test, x_test_fit, y_test_fit, x_test_eval, y_test_eval

def fit_dnn_model(save_model_bool=False, model_path_filename="dnn_model", history_path_filename="history", disable_gpu_training=False, batch_size=32, epochs=5):
    if (disable_gpu_training):
        tf.config.set_visible_devices([], 'GPU')

    x_train, x_test, y_train, y_test, x_test_fit, y_test_fit, x_test_eval, y_test_eval = load_normalized_and_encoded_dataset()

    model.summary()

    print("disable_gpu_training: ", disable_gpu_training)
    print("save_model: ", save_model_bool)
    print("batch_size: ", batch_size)
    print("epochs: ", epochs)

    # Train the model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test_fit, y_test_fit))

    if (save_model_bool):
        print()

        now = formatted_datetime()
        save_model(model, history, model_path_filename + "_" + now, history_path_filename)

    return model, history, x_train, x_test, y_train, y_test, x_test_fit, y_test_fit, x_test_eval, y_test_eval


if __name__ == "__main__":

    print('Tensorflow ', tf.__version__)

    is_gpu_supported()

    save_model_bool = True
    model_path_filename = "dnn_model"
    history_path_filename = "history"
    disable_gpu_training = True
    batch_size = 64
    epochs = 5
    load_model_bool = True

    if(load_model_bool):
        model_load_filename = "dnn_model_2023-07-05_15_34_26_749790"

        model, history = load_model(model_load_filename, history_path_filename)

        x_train, x_test, y_train, y_test, x_test_fit, y_test_fit, x_test_eval, y_test_eval = load_normalized_and_encoded_dataset()

    else:
        model, history, x_train, x_test, y_train, y_test, x_test_fit, y_test_fit, x_test_eval, y_test_eval = \
            fit_dnn_model(save_model_bool=save_model_bool, model_path_filename=model_path_filename,
            history_path_filename=history_path_filename, disable_gpu_training=disable_gpu_training,
            batch_size=batch_size, epochs=epochs)

    # Select a random test sample
    sample_idx = np.random.randint(0, len(x_test_eval))
    x_sample = x_test_eval[sample_idx:sample_idx + 1]
    y_sample = y_test_eval[sample_idx:sample_idx + 1]
    #
    # Generate adversarial example for the selected sample
    # x_adv_sample = generate_adversarial_examples(model, x_sample, y_sample)
    #
    # # Evaluate the model on the adversarial example
    # _, acc = model.evaluate(x_adv_sample, y_sample)
    # print(f'Accuracy on adversarial example: {acc}')
    #
    # Evaluate the model on the original sample
    acc = model.evaluate(x_sample, y_sample)
    print(f'Accuracy on original sample: {acc}')