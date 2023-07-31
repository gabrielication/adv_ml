from util import *

from fix_cw_l2 import carlini_wagner_l2

# Preprocessing function
def preprocess_img(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def filter_by_class(target_class_idx):
    def filter_class(image, label):
        return label == target_class_idx
    return filter_class

def make_cw_targeted_attack(model_path_filename, history_path_filename, target_class, one_hot_depth=10):

    model, history = load_model(model_path_filename, history_path_filename)

    # Loading the imagenette dataset
    ds = tfds.load('imagenette/320px', split='validation', as_supervised=True)

    # Preprocess the dataset
    ds = ds.map(
        preprocess_img, num_parallel_calls=tf.data.AUTOTUNE).batch(1)

    # We have to one-hot encode the label. The depth is the size of the vector
    # We have 10 classes in Imagenette then depth = 10

    target_one_hot_enc = tf.one_hot(target_class, depth=one_hot_depth)

    test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
    test_acc_cw = tf.metrics.SparseCategoricalAccuracy()

    # target_class_idx = 5  # replace with your desired class index
    # filtered_ds = ds.filter(filter_by_class(target_class_idx))

    # Evaluate on clean and adversarial data

    for x, y in ds:
        # print("pred")
        # y_pred = model(x)
        # test_acc_clean(y, y_pred)
        #
        # # BUG!!!! https://github.com/cleverhans-lab/cleverhans/issues/1205#issuecomment-1028411235
        print("cw")
        x_cw = carlini_wagner_l2(model, x, clip_min=-1.0, clip_max=1.0)
        y_pred_fgm = model(x_cw)
        test_acc_cw(y, y_pred_fgm)

        break


if __name__ == "__main__":
    print('Tensorflow ', tf.__version__)

    # Where to save/load the fitted model and its history file
    model_path_filename = "tf_saved_models/no_backup/resnet50_imagenette_2023-07-28_16_34_42_298453"
    history_path_filename = "classification_history_model"

    adversarial_images = make_cw_targeted_attack(model_path_filename, history_path_filename, 1)


    print()