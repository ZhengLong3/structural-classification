import torch

import utils

from gaussian_model import GaussianModel

gaussian = GaussianModel(3)
gaussian.load_ply("output/2plane_parallel.ply")

gaussian.filter_flat_gaussians(3)
labels = utils.spectral_clustering_k(gaussian.get_adjacency_matrix(), 3, 2)
gaussian.colour_by_label(labels)
gaussian.save_ply("output/2plane_parallel_spec_cluster.ply")
