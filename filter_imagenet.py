# Define the dictionary
dict_similar = {}

import pickle

# Open the file and read the data
with open('semantically_similar.txt', 'r') as f:
    for line in f:
        # Split each line into a key-value pair
        key, value = line.split()

        print(key,value)

        # Add the key-value pair to the dictionary, stripping any extra whitespace
        dict_similar[key] = value

# Define the dictionary
dict_images = {}

# Open the file and read the data
with open('val.txt', 'r') as f:
    for line in f:
        # Split each line into a key-value pair
        value, key = line.split()

        # print(key,value)

        if key in dict_images:
            dict_images[key].append(value)
        else:
            dict_images[key] = [value]

# print(dict_images)

images_keys = list(dict_images.keys())
dict_subimages = {}

for key in images_keys:
    dict_subimages[key] = dict_images[key][:10]

print(dict_subimages)

with open('dict_subimages.pickle', 'wb') as handle:
    pickle.dump(dict_subimages, handle, protocol=pickle.HIGHEST_PROTOCOL)


