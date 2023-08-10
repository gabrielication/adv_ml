import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

l0_dict = None
l1_dict = None
l2_dict = None
linf_dict = None

# Replace 'your_file_path.pkl' with the path to your pickle file.
with open('ldicts/l0_norm_dict.pickle', 'rb') as file:
    l0_dict = pickle.load(file)

# Replace 'your_file_path.pkl' with the path to your pickle file.
with open('ldicts/l1_norm_dict.pickle', 'rb') as file:
    l1_dict = pickle.load(file)

# Replace 'your_file_path.pkl' with the path to your pickle file.
with open('ldicts/l2_norm_dict.pickle', 'rb') as file:
    l2_dict = pickle.load(file)

# Replace 'your_file_path.pkl' with the path to your pickle file.
with open('ldicts/linf_norm_dict.pickle', 'rb') as file:
    linf_dict = pickle.load(file)

dicts_list = [l0_dict, l1_dict, l2_dict, linf_dict]

count = 0

for lnorm_dict in dicts_list:

    # Replace these with your data and loop variables
    rows = 10
    cols = 10

    # Create an empty matrix filled with zeros
    matrix = np.zeros((rows, cols), dtype=float)

    print("Converting dict %d to np matrix..." % count)
    # Example: Populating the matrix with values using two for loops
    for i in range(rows):
        for j in range(cols):
            # Replace this line with the logic to fill the matrix based on your data
            matrix[i, j] = lnorm_dict[i][j][0]

    # Set the diagonal of the matrix to 0.0
    np.fill_diagonal(matrix, 0.0)

    # Create the labels
    row_labels = ['tiger', 'zebra', 'gorilla', 'gibbon', 'canoe', 'gondola', 'h-disk', 'sc-bus', 'tramcar', 'volcano']
    col_labels = ['tiger', 'zebra', 'gorilla', 'gibbon', 'canoe', 'gondola', 'h-disk', 'sc-bus', 'tramcar', 'volcano']

    filtered = ['gorilla','gondola']

    # col_labels = ['zebra','sc-bus']
    # Assuming your matrix is named m
    # m = np.array(....)

    # # Add the small constant to avoid issues with log scale
    # epsilon = 1e-10
    # column1 += epsilon
    # column2 += epsilon

    # Calculate the percentage difference
    # percentage_difference = (np.abs(column2 - column1) / ((column1 + column2) / 2)) * 100

    plt.figure(figsize=(14, 8))

    # Plotting each row
    for idx, row_label in enumerate(row_labels):
        if row_label in filtered:
            row_data = matrix[idx, :]
            plt.plot(col_labels, row_data, marker='o', label=row_label)

            # Annotate points for the current row
            for i, val in enumerate(row_data):
                plt.text(i, val, f"{val:.5f}", ha='center', va='bottom')

    if count == 3:
        nstr = "inf"
    else:
        nstr = str(count)

    plt.title('L' + nstr + ' Perturbation Magnitudes for Different Classes')
    plt.xlabel('Target Classes')
    plt.ylabel('Perturbation Magnitude')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
    plt.tight_layout()  # Adjust layout for better visualization

    ax = plt.gca()
    ax.set_yscale('symlog')

    # Set the y limit to the minimum non-zero value from the matrix
    min_value = np.min(matrix[matrix > 0])
    ax.set_ylim(bottom=0)

    plt.show()

    count += 1  # Increment the count after processing each dictionary
