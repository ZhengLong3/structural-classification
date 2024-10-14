# for type hinting of GaussianModel
from __future__ import annotations

import os

import torch
import numpy as np

from plyfile import PlyData, PlyElement
from torch import nn

from utils import quarternion_to_matrix, tensor_to_csv


DEVICE = "cpu"
COLOURS = [[0, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
USEFUL_ATTRIBUTES = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]

# extracted from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py
class GaussianModel:
    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.float, device=DEVICE)
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device=DEVICE).transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device=DEVICE).transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacities, dtype=torch.float, device=DEVICE)
        self._scaling = torch.tensor(scales, dtype=torch.float, device=DEVICE)
        self._rotation = torch.tensor(rots, dtype=torch.float, device=DEVICE)

        self.active_sh_degree = self.max_sh_degree


    def print_gaussian_shapes(self) -> None:
        """
        Prints out the shapes of the various properties of the gaussian splatting models. Used for diagnostics.
        """

        print(f"Number of gaussian splats: {self._xyz.shape[0]}")
        print(f"Shape of _xyz: {self._xyz.shape}")
        print(f"Shape of _features_dc: {self._features_dc.shape}")
        print(f"Shape of _features_rest: {self._features_rest.shape}")
        print(f"Shape of _opacity: {self._opacity.shape}")
        print(f"Shape of _scaling: {self._scaling.shape}")
        print(f"Shape of _rotation: {self._rotation.shape}")


    def filter_flat_gaussians(self, threshold: float = 4) -> None:
        """
        Modifies the gaussian model to only include gaussians which are "flat" enough.

        Parameters:
            threshold: Sets how much lower the minimum scale needs to be compared to the other scales. The scaling seems to be log_2 scales, so the default of 4 requires the thinnest dimension to be 2^4 = 16 times less than the maximum scale.
        """

        top_2, _ = self._scaling.topk(2, dim=1)
        second_scaling, _ = torch.min(top_2, dim=1, keepdim=True)
        min_scaling, _ = torch.min(self._scaling, dim=1, keepdim=True)
        flat_mask = (second_scaling - min_scaling > threshold).squeeze()
        self.filter_gaussian_mask(flat_mask)


    def filter_gaussian_mask(self, mask: torch.Tensor) -> None:
        """
        Filters and modifies the gaussian using a given mask.

        Parameters:
            mask: Tensor of size (n) of booleans or truthy/falsy values to be used as a mask
        """

        for attribute in USEFUL_ATTRIBUTES:
            setattr(self, attribute, getattr(self, attribute)[mask])


    def get_normal_vectors(self) -> torch.Tensor:
        """
        Returns a tensor with the normal vector for each gaussian splat.
        
        Returns:
            torch.Tensor: a tensor of size (n, 3) containing the normal vectors of the corresponding gaussian splat
        """

        VECTORS = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        min_index = torch.argmin(self._scaling, dim=1)
        normal_axes = VECTORS.index_select(0, min_index).unsqueeze(2)
        rotation_matrices = quarternion_to_matrix(self._rotation)
        return torch.matmul(rotation_matrices, normal_axes).squeeze()
    

    def get_adjacency_matrix(self, threshold: float = 0.9, eps: float = 1e-5) -> torch.Tensor:
        """
        Returns the adjacency matrix of the splats for spectral clustering. The diagonals are 0 and the values are inverse distance if the normal vectors have a cosine similarity above a certain threshold.

        Parameters:
            threshold: Minimum cosine similarity before using inverse distance
            eps: Minimum distance when taking inverse distance

        Returns:
            torch.Tensor: A tensor of size (n, n) with diagonals 0 and inverse distance if normals are close enough
        """

        pairwise_distance = torch.cdist(self._xyz, self._xyz)
        pairwise_distance[pairwise_distance < eps] = eps
        inverse_distance = 1.0 / pairwise_distance
        inverse_distance.fill_diagonal_(0)
        normal_vectors = self.get_normal_vectors()
        cosine_similarity = torch.abs(torch.matmul(normal_vectors, normal_vectors.T))
        mask = cosine_similarity > threshold
        return inverse_distance * mask.int().float()

    
    def merge_gaussians(self, other: GaussianModel) -> GaussianModel:
        """
        Modifies the gaussian model by merging another gaussian model's splats to itself.

        Parameters:
            other: The other gaussian model to merge. It will not be modified

        Returns:
            GaussianModel: self
        """

        for attribute in USEFUL_ATTRIBUTES:
            setattr(self, attribute, torch.cat((getattr(self, attribute), getattr(other, attribute)), dim=0))

        return self

    
    def colour_by_label(self, labels: torch.Tensor) -> None:
        """
        Modifies the colours of the gaussian splats according to the provided labels.

        Parameters:
            labels: An integer tensor of size (n) which contains the class of each of the n points
        """

        self._features_dc = torch.tensor(COLOURS).index_select(0, labels).unsqueeze(1)
        self._features_rest = torch.zeros(self._features_rest.shape)
        self._opacity = torch.ones(self._opacity.shape)