import numpy as np

# Define the edges of the graph based on the image
edges = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (5, 7), (6, 7), (7, 8)]

# Initialize an 8x8 matrix of zeros for the adjacency matrix
A = np.zeros((8, 8), dtype=int)

# Populate the adjacency matrix based on the edges
for edge in edges:
    i, j = edge[0] - 1, edge[1] - 1  # Subtract 1 for 0-indexed matrix
    A[i][j] = 1
    A[j][i] = 1  # Undirected graph: the matrix is symmetric

# Degree matrix D
D = np.diag(A.sum(axis=1))

# Laplacian matrix L
L = D - A

# Normalized adjacency matrix A_tilde
degree_matrix_inv_sqrt = np.diag(1 / np.sqrt(A.sum(axis=1)))
A_tilde = degree_matrix_inv_sqrt @ A @ degree_matrix_inv_sqrt

# Normalized Laplacian matrix L_tilde
L_tilde = np.identity(8) - A_tilde

# Formatted output for matrices
def matrix_to_string(matrix):
    return '\n'.join(' '.join(f'{value:8.4f}' for value in row) for row in matrix)

print("Adjacency Matrix A:\n", A)
print("\nNormalized Adjacency Matrix A_tilde:\n", matrix_to_string(A_tilde))
print("\nLaplacian Matrix L:\n", matrix_to_string(L))
print("\nNormalized Laplacian Matrix L_tilde:\n", matrix_to_string(L_tilde))