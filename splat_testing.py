import torch

import utils

from gaussian_model import GaussianModel

gaussian = GaussianModel(3)
gaussian.load_ply("data/cuboid_edited.ply")

utils.filter_flat_gaussians(gaussian, 3)
print(utils.get_normal_vectors(gaussian))