from util import *

from fix_cw_l2 import carlini_wagner_l2

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

# Preprocessing function
def preprocess_img(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label
def make_cw_targeted_attack(model_path_filename, history_path_filename, target_class, one_hot_depth=1000):
    base_model = tf.keras.applications.MobileNetV2(include_top=True, weights="imagenet", input_shape=(224, 224, 3))
    base_model.trainable = False

    model = base_model

    # Loading the imagenette dataset
    ds = tfds.load('imagenette/320px', split='validation', as_supervised=True)

    # Preprocess the dataset
    ds = ds.map(
        preprocess_img, num_parallel_calls=tf.data.AUTOTUNE).batch(1)

    # We have to one-hot encode the label. The depth is the size of the vector
    # We have 10 classes in Imagenette then depth = 10

    test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
    test_acc_cw = tf.metrics.SparseCategoricalAccuracy()

    for x, y in ds:
        print("pred")
        y_pred = model(x)
        test_acc_clean(y, y_pred)

        _, image_class, class_confidence = get_imagenet_label(y_pred.numpy())

        print(image_class, class_confidence)

        batch_size = x.shape[0]  # assuming x is the batch of images
        target_one_hot_enc = tf.one_hot(target_class, depth=one_hot_depth)
        target_one_hot_enc = tf.repeat(target_one_hot_enc[None, :], batch_size, axis=0)

        # BUG!!!! https://github.com/cleverhans-lab/cleverhans/issues/1205#issuecomment-1028411235
        print("cw")
        x_cw = carlini_wagner_l2(model, x, clip_min=-1.0, clip_max=1.0, targeted=True, y=target_one_hot_enc)
        y_pred_fgm = model(x_cw)

        _, image_class, class_confidence = get_imagenet_label(y_pred_fgm.numpy())

        print(image_class, class_confidence)

        test_acc_cw(y, y_pred_fgm)

        break


if __name__ == "__main__":
    print('Tensorflow ', tf.__version__)

    # Where to save/load the fitted model and its history file
    model_path_filename = "tf_saved_models/mobilenetv2_imagenette_2023-07-31_16_57_19_444119"
    history_path_filename = "classification_history_model"

    gpu_id = 1

    choose_gpu(gpu_id=gpu_id)

    target_class = 1

    adversarial_images = make_cw_targeted_attack(model_path_filename, history_path_filename, target_class)

    print()