from utils.util import *

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

'''
useful links:

https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/keras_example.ipynb#scrollTo=XWqxdmS1NLKA
https://www.kaggle.com/code/kutaykutlu/resnet50-transfer-learning-cifar-10-beginner
https://stackabuse.com/split-train-test-and-validation-sets-with-tensorflow-datasets-tfds/
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/benchmarks/resnet50/resnet50.py
'''


# Preprocessing function
def preprocess_img(image, label):
    '''
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/preprocess_input
    MobileNet is the only that normalizes pixels in a certain range, i.e., [-1,1]
    ResNet does a caffe preprocessing which is much more complicated to attack
    '''
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    # image /= 255.0  # normalize to [0,1] range
    return image, label


def load_mobilenet_model(summary=False, include_top=False, weights="imagenet", model_trainable=True):
    print("weights: ", weights)
    print("include_top: ", include_top)
    print("model_trainable: ", model_trainable)

    '''
        We are relying on Transfer learning, using a pretrained ResNet50 model, and readapt it
        in order to be used in our use case

        "include_top" refers to the final layer or set of layers of the model,
        which are typically fully connected layers used for classification, so we are
        excluding them since we have a different dataset to classify

        with include_top=False, it doesn't know the exact dimensions of the output tensor,
        because it depends on the input shape, which can be any size. we need to specify the input shape
        when loading the ResNet50 model.
    '''

    base_model = tf.keras.applications.MobileNetV2(include_top=include_top, weights=weights, input_shape=(224, 224, 3))
    base_model.trainable = model_trainable

    '''
    In the official tensorflow implementation, ResNet50's output layer is composed only of 
    Flatten layer and a Dense layer with the number of classes. In Imagenette we have only 10
    classes, so we do that. Imagenette is still ImageNet. It's a subset but of the original
    images. 

    If the new dataset is small and similar to the original dataset, it's a good 
    practice to only train your classifier (the top, i.e., output, layers you add on the model), keeping 
    the base model frozen. This is because with a small dataset, there's a high risk of 
    overfitting if you try to train the entire model. Since the tasks are similar, the 
    features that the base model has learned could already be very useful.

    If the new dataset is large and similar to the original dataset, you can try fine-tuning 
    the entire model. You start with the pre-trained weights, and the model will adapt those 
    weights to the new task. This can give you better performance since the model has more 
    capacity to learn, and it's not likely to overfit given the large amount of data.

    Imagenet it's the same but smaller. So we can set model_trainable to false to froze the
    original weights and train only the new output. The final Dense layer that we added is 
    initialized with random weights and biases. It doesn't yet know how to make the correct 
    predictions for your specific task (classifying into 10 classes in the case of Imagenette).
    We do training just for it, and the task will also be a lot faster.
    '''

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
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


def load_imagenette():
    print("loading Imagenette...")

    (ds_train, ds_val, ds_test), ds_info = tfds.load('imagenette/320px',
                                                     split=["train", "validation[80%:]", "validation[:80%]"],
                                                     shuffle_files=True, with_info=True,
                                                     as_supervised=True)

    '''
    This line applies the normalize_img function to each element in the ds_train dataset.
    The map function applies a given function to each element in the dataset. num_parallel_
    calls=tf.data.AUTOTUNE tells TensorFlow to choose an optimal number of threads to run
    the function in parallel, improving loading speed.
    '''

    ds_train = ds_train.map(
        preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

    ds_val = ds_val.map(
        preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

    ds_test = ds_test.map(
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

    ds_val = ds_val.batch(128)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info

def load_custom_imagenet(directory, img_height, img_width, batch_size):
    print("loading custom imagenet...")

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

    train_dataset = train_dataset.map(preprocess_img)
    val_dataset = val_dataset.map(preprocess_img)

    ds_train = train_dataset
    ds_val = val_dataset

    return ds_train, ds_val

def evaluate_model_after_training(model, ds_test):
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(ds_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")


def fit_resnet_and_save_model(model_path_filename, history_path_filename, summary=True, epochs=5, save_weights=False,
                              base_model_trainable=True):
    print("epochs: ", epochs)
    print("save_weights: ", save_weights)

    model = load_mobilenet_model(summary=summary, model_trainable=base_model_trainable)

    # ds_train, ds_val, ds_test, ds_info = load_imagenette()

    ds_train, ds_val = load_custom_imagenet("ds/", 224, 224, 128)
    ds_test = None

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
        validation_data=ds_val,
        callbacks=[early_stopping]
    )

    save_model(model, history, model_path_filename, history_path_filename, save_weights=save_weights)

    return model, history, ds_test


if __name__ == "__main__":
    print('Tensorflow ', tf.__version__)

    # Where to save/load the fitted model and its history file
    model_path_filename = "tf_saved_models/mobilenetv2_imagenette"
    history_path_filename = "classification_history_model"

    epochs = 100

    gpu_id = 0

    save_weights = False

    base_model_trainable = False

    choose_gpu(gpu_id=gpu_id)

    model, history, ds_test = fit_resnet_and_save_model(model_path_filename, history_path_filename, summary=True,
                                                        epochs=epochs, save_weights=save_weights,
                                                        base_model_trainable=base_model_trainable)

    evaluate_model_after_training(model, ds_test)