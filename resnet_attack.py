from util import *

from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2

if __name__ == "__main__":
    print('Tensorflow ', tf.__version__)

    # Where to save/load the fitted model and its history file
    model_path_filename = "tf_saved_models/resnet50_cifar10_weights_2023-07-27_16_52_38_564585"
    history_path_filename = "classification_history_model"

    model, history = load_model(model_path_filename, history_path_filename)

    print()