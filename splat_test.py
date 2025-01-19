import torch

import utils

from gaussian_model import GaussianModel


gaussian = GaussianModel(3)
gaussian.load_ply("data/book_close_flat.ply")
utils.tensor_to_csv(gaussian._scaling, "./output/scalings")

'''
# gaussian.filter_flat_gaussians(3)
gaussian.filter_opaque_gaussians(0.3)
adjacency_matrix = gaussian.get_adjacency_matrix(threshold=0.97)
zero_mask = adjacency_matrix == 0
top_k_values, top_k_idx = torch.topk(adjacency_matrix, 4, sorted=False)
spectral_matrix = torch.zeros(adjacency_matrix.shape)
spectral_matrix[torch.arange(spectral_matrix.shape[0]), top_k_idx.t()] = 1
spectral_matrix[zero_mask] = 0
# threshold = torch.quantile(adjacency_matrix.reshape((-1,)), 0.95)
# spectral_matrix = adjacency_matrix >= threshold
labels = utils.spectral_clustering_zeros(spectral_matrix.float(), 1e-2)
# labels = utils.spectral_clustering_k(spectral_matrix.float(), 2, 2)
# utils.tensor_to_csv(labels, "output/labels.csv")
gaussian.colour_by_label(labels)
gaussian.save_ply("output/book_close_flat_spec.ply")
'''
