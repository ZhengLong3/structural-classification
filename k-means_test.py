import torch

from gaussian_model import GaussianModel
from utils import quarternion_to_matrix, k_means

gaussian = GaussianModel(3)
gaussian.load_ply("./data/cuboid_edited.ply")

features = torch.concat((gaussian._xyz, gaussian._scaling, gaussian._rotation), dim=1)
# print(features.shape)

labels = k_means(features, 5)

# new_features_dc = torch.tensor(COLOURS).index_select(0, labels)
# gaussian._features_dc = new_features_dc.unsqueeze(1)

# new_features_rest = torch.zeros(gaussian._features_rest.shape)
# gaussian._features_rest = new_features_rest

# new_opacity = torch.ones(gaussian._opacity.shape)
# gaussian._opacity = new_opacity

# torch.set_printoptions(profile="full")
# print(gaussian._rotation)
print(torch.nn.functional.normalize(gaussian._rotation, p=2, dim=1))
print(quarternion_to_matrix(torch.nn.functional.normalize(gaussian._rotation, p=2, dim=1)))

# gaussian._scaling = -gaussian._scaling

# gaussian.save_ply("./output/cubiod_abs_scaling.ply")