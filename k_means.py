import torch

from gaussian_model import GaussianModel

NUM_CLUSTERS = 2
MAX_ITER = 1000

gaussian = GaussianModel(3)
gaussian.load_ply("./2_plane_3_edited.ply")

features = gaussian._xyz

# randomly choose centers from input
perm = torch.randperm(features.shape[0])
centroids = features[perm[:NUM_CLUSTERS]]

labels = torch.zeros(features.shape[0], dtype=torch.uint8)
repeated_features = features.unsqueeze(1).repeat(1, NUM_CLUSTERS, 1)

# k-means
for _ in range(MAX_ITER):
    old_centroids = centroids

    # assign closest
    difference = repeated_features - centroids
    norm = torch.linalg.norm(difference, ord=2, dim=-1)
    labels = torch.argmin(norm, dim=1)

    # get new centroids
    # taken from https://stackoverflow.com/a/56155805
    M = torch.zeros(labels.max() + 1, len(features))
    M[labels, torch.arange(len(features))] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)
    centroids = torch.mm(M, features)

    if torch.mean(torch.linalg.norm(centroids - old_centroids, ord=2, dim=1)) < 0.00001:
        break

new_features_dc = labels.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3)
gaussian._features_dc = new_features_dc

new_features_rest = labels.unsqueeze(-1).unsqueeze(-1).repeat(1, 15, 3)
gaussian._features_rest = new_features_rest

gaussian.save_ply("./2_plane_3_clustered.ply")