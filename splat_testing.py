import torch

import utils

from gaussian_model import GaussianModel

gaussian = GaussianModel(3)
gaussian.load_ply("data/book.ply")

gaussian.filter_flat_gaussians(4)
gaussian.filter_opaque_gaussians(0.50)
adjacency_matrix = gaussian.get_adjacency_matrix(threshold=0.97)
flattened_adj_mat = torch.flatten(adjacency_matrix)
adj_mat_sample = flattened_adj_mat[torch.randperm(len(flattened_adj_mat))[:200]]
distance_threshold = torch.quantile(adj_mat_sample, 0.95, dim=0)
adjacency_matrix = (adjacency_matrix >= distance_threshold).float()
labels = utils.spectral_clustering_zeros(adjacency_matrix, 1)
utils.tensor_to_csv(labels, "output/labels.csv")
gaussian.colour_by_label(labels)
gaussian.save_ply("output/book_spec_cluster.ply")
