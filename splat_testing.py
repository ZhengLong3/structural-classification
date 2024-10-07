import torch

import utils

from gaussian_model import GaussianModel

gaussian = GaussianModel(3)
gaussian.load_ply("data/cuboid_edited.ply")

gaussian.filter_flat_gaussians(3)
print(gaussian.get_normal_vectors())