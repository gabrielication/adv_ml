# Define the dictionary
dict_similar = {}

import pickle

# Open the file and read the data
with open('semantically_similar.txt', 'r') as f:
    for line in f:
        # Split each line into a key-value pair
        value1, key, value2 = line.split()

        print(key,value1,value2)

        # Add the key-value pair to the dictionary, stripping any extra whitespace
        dict_similar[key] = [value1, value2]

# Define the dictionary
dict_images = {}

# Open the file and read the data
with open('val.txt', 'r') as f:
    for line in f:
        # Split each line into a key-value pair
        value, key = line.split()

        # print(key,value)

        if key in dict_similar:
            similar_key = dict_similar[key][1]
            if similar_key in dict_images:
                dict_images[similar_key].append(value)
            else:
                dict_images[similar_key] = [value]

# print(dict_images)

images_keys = list(dict_images.keys())
dict_subimages = {}

for key in images_keys:
    dict_subimages[key] = dict_images[key][:]

print(dict_subimages)

# with open('../dict_subimages.pickle', 'wb') as handle:
#     pickle.dump(dict_subimages, handle, protocol=pickle.HIGHEST_PROTOCOL)

for key in dict_subimages:
    import os
    import shutil

    print(key)

    path_src = "../ILSVRC2012_img_val/"
    path_des = "../ds/"+key

    os.makedirs(path_des, exist_ok=True)

    images = dict_subimages[key]

    # Loop over the filenames and copy each file
    for filename in images:
        src_file = os.path.join(path_src, filename)
        dst_file = os.path.join(path_des, filename)
        shutil.copy(src_file, dst_file)




