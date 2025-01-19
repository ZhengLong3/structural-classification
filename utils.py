import numpy as np
import torch

COLOURS = [[0, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]

def quarternion_to_matrix(quarternions: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of quaternions into a batch of rotation matrices.

    Parameters:
        quarternions: A tensor with shape (..., 4) containing the quarternions

    Returns:
        torch.Tensor: A tensor with shape (..., 3, 3) containing the rotation matrices
    """

    quarternions = torch.nn.functional.normalize(quarternions, p=2, dim=1)
    real = quarternions[:, 0]
    i = quarternions[:, 1]
    j = quarternions[:, 2]
    k = quarternions[:, 3]
    output_00 = torch.square(real) + torch.square(i) - torch.square(j) - torch.square(k)
    output_01 = 2 * (i * j - real * k)
    output_02 = 2 * (i * k + real * j)
    output_10 = 2 * (i * j + real * k)
    output_11 = torch.square(real) - torch.square(i) + torch.square(j) - torch.square(k)
    output_12 = 2 * (j * k - real * i)
    output_20 = 2 * (i * k - real * j)
    output_21 = 2 * (j * k + real * i)
    output_22 = torch.square(real) - torch.square(i) - torch.square(j) + torch.square(k)
    row_0 = torch.stack((output_00, output_01, output_02), dim=1)
    row_1 = torch.stack((output_10, output_11, output_12), dim=1)
    row_2 = torch.stack((output_20, output_21, output_22), dim=1)
    return torch.stack((row_0, row_1, row_2), dim=1)


def k_means(features: torch.Tensor, num_clusters: int, max_iter: int = 1000, threshold: float = 0.00001, ord: int = 2) -> torch.Tensor:
    """
    Applies k-means clustering to a tensor of points.

    Parameters:
        features: A tensor of shape (n, m), with n points of m dimensions
        num_clusters: Number of clusters to cluster the points into
        max_iter: Maximum number of iterations for k-means clustering (default: 1000)
        threshold: Average difference between new centroids and old centroids before terminating (default: 0.00001)
        ord: Type of L_n norm to use (default: 2)

    Returns
        torch.Tensor: A tensor containing labels 0 to num_clusters - 1 for the clustered points.
    """

    # randomly choose centers from input
    perm = torch.randperm(features.shape[0])
    centroids = features[perm[:num_clusters]]

    labels = torch.zeros(features.shape[0], dtype=torch.uint8)
    repeated_features = features.unsqueeze(1).repeat(1, num_clusters, 1)

    # k-means
    for _ in range(max_iter):
        old_centroids = centroids

        # assign closest
        difference = repeated_features - centroids
        norm = torch.linalg.norm(difference, ord=ord, dim=-1)
        labels = torch.argmin(norm, dim=1)

        # get new centroids
        # taken from https://stackoverflow.com/a/56155805
        M = torch.zeros(labels.max() + 1, len(features))
        M[labels, torch.arange(len(features))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        centroids = torch.mm(M, features)

        if torch.mean(torch.linalg.norm(centroids - old_centroids, ord=ord, dim=1)) < threshold:
            break
    
    return labels


def spectral_clustering_zeros(adjacency_matrix: torch.Tensor, zero_threshold: float) -> torch.Tensor:
    """
    Does spectral clustering based on zero eigenvalues on an adjacency matrix and returns the labels as a tensor.

    Parameters:
        adjacency_matrix: A tensor of size (n, n) with diagonals 0 for spectral clustering
        zero_threshold: Threshold for an eigenvalue to be considered zero
        num_clusters: Number of clusters when clustering with k-means

    Returns:
        torch.Tensor: A tensor of size (n) containing the labels for each node in the adjacency matrix
    """

    degree_matrix = torch.diag(adjacency_matrix.sum(0))
    graph_lapacian = degree_matrix - adjacency_matrix
    eigen_values, eigen_vectors = torch.linalg.eig(graph_lapacian)
    # tensor_to_csv(eigen_values.real, "output/eig_val.csv")
    # tensor_to_csv(eigen_vectors.real, "output/eig_vec.csv")
    zero_mask = eigen_values.real <= zero_threshold
    num_zeroes = torch.sum(zero_mask)
    print(f"Found {num_zeroes} zero eigenvalues")
    eigen_vectors = eigen_vectors[:, zero_mask].real
    return k_means(eigen_vectors, num_clusters=num_zeroes)


def spectral_clustering_k(adjacency_matrix: torch.Tensor, k: int, num_clusters: int) -> torch.Tensor:
    """
    Does spectral clustering based on smallest k eigenvalues on an adjacency matrix and returns the labels as a tensor.

    Parameters:
        adjacency_matrix: A tensor of size (n, n) with diagonals 0 for spectral clustering
        k: First k smallest eigenvalues to use
        num_clusters: Number of clusters when clustering with k-means

    Returns:
        torch.Tensor: A tensor of size (n) containing the labels for each node in the adjacency matrix
    """

    degree_matrix = torch.diag(adjacency_matrix.sum(0))
    graph_lapacian = degree_matrix - adjacency_matrix
    eigen_values, eigen_vectors = torch.linalg.eig(graph_lapacian)
    # tensor_to_csv(eigen_values.real, "output/eig_val.csv")
    # tensor_to_csv(eigen_vectors.real, "output/eig_vec.csv")
    _, topk_indices = torch.topk(eigen_values.real, k=k, largest=False, sorted=False)
    eigen_vectors = eigen_vectors[:, topk_indices].real
    return k_means(eigen_vectors, num_clusters=num_clusters)


def tensor_to_csv(tensor: torch.Tensor, path: str) -> None:
    """
    Saves a tensor to a csv file.

    Parameters:
        tensor: A tensor to be saved to CSV
        path: A file path to save the CSV file
    """
    np.savetxt(path, tensor.numpy(), delimiter=",")