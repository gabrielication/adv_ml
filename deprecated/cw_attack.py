from utils.util import *

from fix_cw_l2 import carlini_wagner_l2

'''
useful links:

https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py

l2 attack is the most efficient

https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
https://raw.githubusercontent.com/Jing-Luo/GLUCLSON/master/basics/val.txt
https://raw.githubusercontent.com/hujie-frank/SENet/master/ILSVRC2017_val.txt

'''

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

imagenette_labels = {
    0: 'tench',
    1: 'English springer spaniel',
    2: 'cassette player',
    3: 'chainsaw',
    4: 'church',
    5: 'French horn',
    6: 'garbage truck',
    7: 'gas pump',
    8: 'golf ball',
    9: 'parachute',
}

l0_norm_dict = {
    0: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    1: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    2: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    3: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    4: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    5: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    6: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    7: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    8: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    9: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
}
l1_norm_dict = {
    0: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    1: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    2: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    3: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    4: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    5: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    6: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    7: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    8: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    9: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
}
l2_norm_dict = {
    0: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    1: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    2: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    3: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    4: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    5: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    6: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    7: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    8: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    9: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
}
linf_norm_dict = {
    0: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    1: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    2: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    3: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    4: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    5: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    6: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    7: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    8: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
    9: [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]],
}

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

def calculate_l_norm(og_img, adv_img):
    # Compute difference
    diff = tf.abs(og_img - adv_img)

    # L0 norm: count of non-zero elements (number of changed pixels)
    L0_norm = tf.reduce_sum(tf.cast(tf.greater(diff, 0), tf.float32))
    L0_norm = L0_norm.numpy()
    # print("L0 norm:", L0_norm)

    # L1 norm: sum of absolute differences (total amount of change)
    L1_norm = tf.norm(diff, ord=1)
    L1_norm = L1_norm.numpy()
    # print("L1 norm:", L1_norm)

    # L2 norm: Euclidean distance (geometric distance in the image space)
    L2_norm = tf.norm(diff, ord=2)
    L2_norm = L2_norm.numpy()
    # print("L2 norm:", L2_norm)

    # Infinity norm: maximum absolute difference (largest change to any pixel)
    Linf_norm = tf.norm(diff, np.inf)
    Linf_norm = Linf_norm.numpy()
    # print("Infinity norm:", Linf_norm)

    return L0_norm, L1_norm, L2_norm, Linf_norm

def calculate_distance_between_two_sample_imgs(ds):
    global l0_norm_dict
    global l1_norm_dict
    global l2_norm_dict
    global linf_norm_dict

    # Take two images from the dataset
    dataset_iter = iter(ds)
    image1 = next(dataset_iter)[0]
    image2 = next(dataset_iter)[0]

    image1, label1 = preprocess_img(image1, None)
    image2, label2 = preprocess_img(image2, None)

    L0_norm, L1_norm, L2_norm, Linf_norm = calculate_l_norm(image1, image2)

    print(L0_norm, L1_norm, L2_norm, Linf_norm)

def one_hot_encode_int_label(x, target_class, depth=10):
    # We have to one-hot encode the label. The depth is the size of the vector
    # We have 10 classes in Imagenette then depth = 10
    batch_size = x.shape[0]  # assuming x is the batch of images
    target_one_hot_enc = tf.one_hot(target_class, depth=depth)
    target_one_hot_enc = tf.repeat(target_one_hot_enc[None, :], batch_size, axis=0)

    return target_one_hot_enc

def calculate_probabilities_from_logits(y_pred):
    # Use the softmax function to convert logits to probabilities
    probabilities = tf.nn.softmax(y_pred)

    # Change the print options
    np.set_printoptions(precision=4, suppress=True)

    # Get the index of the maximum probability
    index_max_prob = tf.argmax(probabilities[0])

    index_max_prob = index_max_prob.numpy()

    return probabilities, index_max_prob

# Preprocessing function
def preprocess_img(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label
def make_cw_targeted_attack(model_path_filename, history_path_filename, batch_size=32):
    global imagenette_labels
    global l0_norm_dict
    global l1_norm_dict
    global l2_norm_dict
    global linf_norm_dict

    model, history = load_model(model_path_filename, history_path_filename)

    # Loading the imagenette dataset
    ds, info = tfds.load('imagenette/320px', with_info=True, split='validation', as_supervised=True)

    # Preprocess the dataset
    ds = ds.map(
        preprocess_img, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

    keysList = list(imagenette_labels.keys())

    for x,y in ds:
        for target_class in keysList:

            ogl_target = y.numpy()
            adv_target = target_class

            target_one_hot_enc = one_hot_encode_int_label(x, target_class)

            print("CW attack in progress. Might take a while...")
            adv_img_batch = carlini_wagner_l2(model, x, clip_min=-1.0, clip_max=1.0, targeted=True, y=target_one_hot_enc)

            for i in range(x.shape[0]):
                ogl_img = x[i]
                adv_img = adv_img_batch[i]
                # TEST ONLY:
                # adv_img = ogl_img

                L0_norm, L1_norm, L2_norm, Linf_norm = calculate_l_norm(ogl_img, adv_img)

                '''
                ln_norm_dict[ogl_target][target_class] represents what happens at the original class
                ogl_target when we target to attack it and be target_class
                
                ln_norm_dict[ogl_target][target_class][0] has the sum of all L0_norms in respect
                to these classes
                
                ln_norm_dict[ogl_target][target_class][1] its a counter of all images for
                these classes in order to the average later
                '''
                l0_old_avg = l0_norm_dict[ogl_target[i]][target_class][0]
                l1_old_avg = l1_norm_dict[ogl_target[i]][target_class][0]
                l2_old_avg = l2_norm_dict[ogl_target[i]][target_class][0]
                linf_old_avg = linf_norm_dict[ogl_target[i]][target_class][0]

                l0_old_count = l0_norm_dict[ogl_target[i]][target_class][1]
                l1_old_count = l1_norm_dict[ogl_target[i]][target_class][1]
                l2_old_count = l2_norm_dict[ogl_target[i]][target_class][1]
                linf_old_count = linf_norm_dict[ogl_target[i]][target_class][1]

                l0_new_avg = (l0_old_avg * l0_old_count + L0_norm) / (l0_old_count + 1)
                l1_new_avg = (l1_old_avg * l1_old_count + L1_norm) / (l1_old_count + 1)
                l2_new_avg = (l2_old_avg * l2_old_count + L2_norm) / (l2_old_count + 1)
                linf_new_avg = (linf_old_avg * linf_old_count + Linf_norm) / (linf_old_count + 1)

                l0_norm_dict[ogl_target[i]][target_class][0] = l0_new_avg
                l1_norm_dict[ogl_target[i]][target_class][0] = l1_new_avg
                l2_norm_dict[ogl_target[i]][target_class][0] = l2_new_avg
                linf_norm_dict[ogl_target[i]][target_class][0] = linf_new_avg

                l0_norm_dict[ogl_target[i]][target_class][1] += 1
                l1_norm_dict[ogl_target[i]][target_class][1] += 1
                l2_norm_dict[ogl_target[i]][target_class][1] += 1
                linf_norm_dict[ogl_target[i]][target_class][1] += 1

    # Save to disk
    with open('l0_norm_dict.pickle', 'wb') as handle:
        pickle.dump(l0_norm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('l1_norm_dict.pickle', 'wb') as handle:
        pickle.dump(l1_norm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('l2_norm_dict.pickle', 'wb') as handle:
        pickle.dump(l2_norm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('linf_norm_dict.pickle', 'wb') as handle:
        pickle.dump(linf_norm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(l0_norm_dict)


if __name__ == "__main__":
    print('Tensorflow ', tf.__version__)

    # Where to save/load the fitted model and its history file
    model_path_filename = "../tf_saved_models/no_backup/mobilenetv2_imagenette_2023-07-31_16_57_19_444119"
    history_path_filename = "classification_history_model"

    gpu_id = 0

    choose_gpu(gpu_id=gpu_id)

    batch_size = 64

    adversarial_images = make_cw_targeted_attack(model_path_filename, history_path_filename, batch_size=batch_size)

    print()