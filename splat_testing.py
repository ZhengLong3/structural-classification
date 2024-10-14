import torch

import utils

from gaussian_model import GaussianModel

gaussian = GaussianModel(3)
gaussian.load_ply("output/2plane_parallel.ply")

gaussian.filter_flat_gaussians(3)
# print(gaussian.get_normal_vectors())
labels = utils.spectral_clustering(gaussian.get_adjacency_matrix(), 1.0, 2)
utils.tensor_to_csv(labels, "output/labels.csv")
gaussian.colour_by_label(labels)
gaussian.save_ply("./output/2plane_parallel_spec_cluster.ply")
