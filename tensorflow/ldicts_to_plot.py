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

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 20))

    # Set the diagonal of the matrix to 0.0
    np.fill_diagonal(matrix, 0.0)

    # Plot the matrix
    cax = ax.matshow(matrix, cmap='Blues')

    row_labels = ['tiger', 'zebra', 'gorilla', 'gibbon', 'canoe', 'gondola', 'h-disk', 'sc-bus', 'tramcar', 'volcano']
    col_labels = ['tiger', 'zebra', 'gorilla', 'gibbon', 'canoe', 'gondola', 'h-disk', 'sc-bus', 'tramcar', 'volcano']

    # Add grid lines
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)

    # Add values to the matrix with higher precision
    print("Adding values to plot %d..." % count)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, '{:.3e}'.format(matrix[i, j]), ha='center', va='center', color='black', fontweight='bold')

    # Set colorbar
    cbar = fig.colorbar(cax)

    # Set axis labels
    plt.xlabel("Original Classes")
    plt.ylabel("Target Classes")

    if count == 3:
        nstr = "inf"
    else:
        nstr = str(count)

    title = "L"+nstr+" norm"
    filename = "L"+nstr+"_matrix.png"

    # Show the plot
    plt.title(title)

    # Save the plot as an image file (e.g., PNG)
    plt.savefig(filename, dpi=300)  # Change the filename and extension as needed

    print("Plot saved as: ",filename)

    count += 1

    # plt.show()

print()