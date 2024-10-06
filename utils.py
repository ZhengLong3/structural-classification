import torch

from gaussian_model import GaussianModel

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


def print_gaussian_shapes(gaussian: GaussianModel) -> None:
    """
    Prints out the shapes of the various properties of the gaussian splatting models. Used for diagnostics.

    Parameters:
        gaussian: Gaussian model to print
    """

    print(f"Number of gaussian splats: {gaussian._xyz.shape[0]}")
    print(f"Shape of _xyz: {gaussian._xyz.shape}")
    print(f"Shape of _features_dc: {gaussian._features_dc.shape}")
    print(f"Shape of _features_rest: {gaussian._features_rest.shape}")
    print(f"Shape of _opacity: {gaussian._opacity.shape}")
    print(f"Shape of _scaling: {gaussian._scaling.shape}")
    print(f"Shape of _rotation: {gaussian._rotation.shape}")


def filter_flat_gaussians(gaussian: GaussianModel, threshold: float = 4) -> None:
    """
    Modifies the gaussian model to only include gaussians which are "flat" enough.

    Parameters:
        gaussian: Gaussian model to be filtered
        threshold: Sets how much lower the minimum scale needs to be compared to the other scales. The scaling seems to be log_2 scales, so the default of 4 requires the thinnest dimension to be 2^4 = 16 times less than the maximum scale.
    """

    top_2, _ = gaussian._scaling.topk(2, dim=1)
    second_scaling, _ = torch.min(top_2, dim=1, keepdim=True)
    min_scaling, _ = torch.min(gaussian._scaling, dim=1, keepdim=True)
    flat_mask = (second_scaling - min_scaling > threshold).squeeze()
    filter_gaussian_mask(gaussian, flat_mask)


def filter_gaussian_mask(gaussian: GaussianModel, mask: torch.Tensor) -> None:
    """
    Filters and modifies the gaussian using a given mask.

    Parameters:
        gaussian: Gaussian model to be filtered
        mask: Tensor of size (n) of booleans or truthy/falsy values to be used as a mask
    """

    gaussian._xyz = gaussian._xyz[mask]
    gaussian._features_dc = gaussian._features_dc[mask]
    gaussian._features_rest = gaussian._features_rest[mask]
    gaussian._opacity = gaussian._opacity[mask]
    gaussian._scaling = gaussian._scaling[mask]
    gaussian._rotation = gaussian._rotation[mask]


def get_normal_vectors(gaussian: GaussianModel) -> torch.Tensor:
    """
    Returns a tensor with the normal vector for each gaussian splat.

    Parameters:
        gaussian: gaussian model to get normal vectors
    
    Returns:
        torch.Tensor: a tensor of size (n, 3) containing the normal vectors of the corresponding gaussian splat
    """

    VECTORS = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    min_index = torch.argmin(gaussian._scaling, dim=1)
    normal_axes = VECTORS.index_select(0, min_index).unsqueeze(2)
    rotation_matrices = quarternion_to_matrix(gaussian._rotation)
    print(rotation_matrices.shape)
    print(normal_axes.shape)
    return torch.matmul(rotation_matrices, normal_axes)