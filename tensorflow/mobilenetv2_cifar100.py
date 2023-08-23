from utils.util import *

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

def load_cifar100(label_mode='coarse', val_samples=10000, resize_w=96, resize_h=96):
    print('loading cifar100...')
    print('label_mode: ',label_mode)
    print('val_samples: ', val_samples)
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode=label_mode)
    
    # Shuffle the training set (optional but recommended)
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    x_val = x_train[:val_samples]
    y_val = y_train[:val_samples]

    x_train = x_train[val_samples:]
    y_train = y_train[val_samples:]
    
    print("resize to: ",[resize_w, resize_h])
    x_train = tf.image.resize(x_train, [resize_w, resize_h])
    x_test = tf.image.resize(x_test, [resize_w, resize_h])
    x_val = tf.image.resize(x_val, [resize_w, resize_h])

    print("mobilenet_v2.preprocess_input...")
    x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train)
    x_test = tf.keras.applications.mobilenet_v2.preprocess_input(x_test)
    x_val = tf.keras.applications.mobilenet_v2.preprocess_input(x_val)
    
    return x_train, y_train, x_val, y_val, x_test, y_test
    
def load_mobilenetv2_model(summary=False, include_top=False, weights="imagenet", input_shape=(96,96,3), num_classes=20):
    print("loading MobileNetV2 model...")
    
    print("include_top: ",include_top)
    print("weights: ",weights)
    print("input_shape: ",input_shape)
    
    base_model = tf.keras.applications.MobileNetV2(include_top=include_top, weights=weights, input_shape=input_shape)
    
    base_model.trainable = False
    
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes))
    
    optimizer = optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    print("optimizer: ", optimizer.get_config()['name'])
    print("loss: ", loss.get_config()['name'])

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    if(summary):
        model.summary()
        
    return model
        
def fine_train_mobilenetv2_to_cifar(epochs=5, batch_size=32, save_model_bool=True):
    
    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar100()
    
    model = load_mobilenetv2_model()
    
    print("epochs: ", epochs)
    
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
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping]
    )
    
    if save_model_bool:
        save_model(model, history, model_path_filename, history_path_filename, save_weights=save_weights)

    return model, history, x_test, y_test
    
if __name__ == "__main__":
    print('Tensorflow ', tf.__version__)

    # Where to save/load the fitted model and its history file
    # model_path_filename = "tf_saved_models/mobilenetv2_imagenette"
    # history_path_filename = "classification_history_model"

    epochs = 100

    gpu_id = 0

    choose_gpu(gpu_id=gpu_id)
    
    model, history, x_test, y_test = fine_train_mobilenetv2_to_cifar(epochs=epochs, save_model_bool=False)
    
    
    
    
    
    
    
    
    
    

