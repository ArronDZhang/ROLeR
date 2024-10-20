import numpy as np
import matplotlib.pyplot as plt

def svd_reduction_and_visualization(matrix):
    """
    Apply SVD factorization to the given matrix and reduce its dimensions to (a, 2).
    Then visualize the reduced matrix using a scatter plot.
    
    :param matrix: A 2D numpy array of shape (a, b) with b > 2.
    :return: The reduced matrix of shape (a, 2).
    """
    # Apply SVD
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)

    # Reduce the dimensions to 2
    reduced_matrix = np.dot(U[:, :2], np.diag(S[:2]))

    # Visualization
    plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('2D Visualization of the Reduced Data')
    # plt.show()
    plt.savefig('Visualization of the State Emb.png')
    plt.close()

    # return reduced_matrix

# # Example usage
# # Create a random matrix of shape (a, b) where b > 2
# a, b = 100, 5  # Example dimensions
# matrix = np.random.random((a, b))

# # Apply SVD reduction and visualization
# reduced_matrix = svd_reduction_and_visualization(matrix)

import pandas as pd

def svd_reduction_and_visualized_with_color(matrix, dataframe, color_column):
    """
    Apply SVD factorization to the given matrix, reduce its dimensions to (a, 2),
    and visualize the reduced matrix using a scatter plot colored by values from a dataframe.

    :param matrix: A 2D numpy array of shape (a, b) with b > 2.
    :param dataframe: A pandas dataframe of shape (a, 2) with columns ["count", "other"].
    :return: The reduced matrix of shape (a, 2).
    """
    # Apply SVD
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)

    # Reduce the dimensions to 2
    reduced_matrix = np.dot(U[:, :2], np.diag(S[:2]))

    # Visualization
    plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c=dataframe[color_column], cmap='magma_r', marker='.')
    plt.colorbar(label=color_column)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Balance the axes
    max_range = max(np.abs(reduced_matrix).max(), np.abs(reduced_matrix).max())
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    # Center the origin
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.grid(True)


    plt.title('2D Visualization of the Reduced Data with Color Coding')
    plt.show()
    plt.close()

    # return reduced_matrix

# Example usage
# Assuming the matrix is the same as before
# Create an example dataframe
# df = pd.DataFrame({
#     'count': np.random.randint(1, 100, size=a),  # Random counts for illustration
#     'other': np.random.random(size=a)  # Some other random data
# })

# # Apply SVD reduction and visualization with color coding
# reduced_matrix_with_color = svd_reduction_and_visualized_with_color(matrix, df)

def just_svd(matrix):

    U, S, VT = np.linalg.svd(matrix, full_matrices=False)

    # Reduce the dimensions to 2
    reduced_matrix = np.dot(U[:, :2], np.diag(S[:2]))
    return reduced_matrix
