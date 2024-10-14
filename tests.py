import math

import torch

from typing import Tuple

from gaussian_model import GaussianModel

def create_test_gaussians(output_path: str) -> None:
    """
    Creates a test gaussian splat model to the output path.

    Parameters:
        output_path: string path of the output .ply file
    """

    gaussians = GaussianModel(3)

    # At two opposite corners of a 5x5x5 cube
    gaussians._xyz = torch.Tensor([[0, 0, 0], [5, 5, 5]])

    # Flat gaussians along the y axis
    gaussians._scaling = torch.Tensor([[-1, -10, -1], [-1, -10, -1]])

    # This rotation is a 90 degree clockwise rotation around the z axis
    gaussians._rotation = torch.Tensor([[0.7071068, 0, 0, 0.7071068], [0.7071068, 0, 0, 0.7071068]])

    # This is for the colour of the splats, red and green respectively
    gaussians._features_dc = torch.Tensor([[[1, 0, 0]], [[0, 1, 0]]])

    # zero so colour does not change depending on view angle
    gaussians._features_rest = torch.zeros((2, 15, 3))
    gaussians._opacity = torch.Tensor([1, 1]).reshape((-1, 1))

    gaussians.save_ply(output_path)


def create_plane_gaussians(center: Tuple[float, float, float], other: Tuple[float, float, float], normal: Tuple[float, float, float], size: float, num_splats: int) -> GaussianModel:
    """
    Creates a gaussian model containing splats in a square plane with a defined center, normal vector and side length.

    Parameters:
        center: Tuple of three floats for the center of the plane
        other: Tuple of three floats for the direction one side of the square should face
        normal: Tuple of three floats for the normal vector of the plane
        size: Side length of the square plane
        num_splats: Number of gaussian splats per side. Please make it larger than 1

    Returns:
        GaussianModel: gaussian model containing the splats of the plane
    """

    total_splats = num_splats * num_splats

    center = torch.Tensor(center)
    other = torch.Tensor(other)
    normal = torch.nn.functional.normalize(torch.Tensor(normal), p=2, dim=0)
    
    # project other orthogonal to normal
    other = other - torch.dot(other, normal) / torch.linalg.norm(other) * normal
    other = torch.nn.functional.normalize(torch.Tensor(other), p=2, dim=0)
    
    # create grid of points
    other2 = torch.cross(other, normal, dim=0)
    corner = center - other * size / 2 - other2 * size / 2
    other = other * size / (num_splats - 1)
    other2 = other2 * size / (num_splats - 1)
    cartesian = torch.cartesian_prod(torch.arange(0, num_splats), torch.arange(0, num_splats)).unsqueeze(dim=-1)
    positions = corner + cartesian[:, 0, :] * other + cartesian[:, 1, :] * other2
    
    # scaling of the splats
    scaling = torch.Tensor([[-12, math.log2(size / num_splats) + 1, math.log2(size / num_splats) + 1] for _ in range(total_splats)])

    # colour and opacity of the splats
    features_dc = torch.rand((total_splats, 1, 3))
    features_rest = torch.zeros((total_splats, 15, 3))
    opacity = torch.ones((total_splats, 1))

    # rotation of splats
    if torch.dot(torch.Tensor((1.0, 0.0, 0.0)), normal) > 0.999999:
        quarternion = torch.Tensor([1.0, 0.0, 0.0, 0.0])
    elif torch.dot(torch.Tensor((1.0, 0.0, 0.0)), normal) < -0.999999:
        quarternion = torch.Tensor([1.0, 0.0, 0.0, 0.0])
    else:
        complex_quarternion = torch.cross(torch.Tensor((1.0, 0.0, 0.0)), normal)
        quarternion = torch.concat(((torch.dot(torch.Tensor((1.0, 0.0, 0.0)), normal) + 1).reshape((1,)), complex_quarternion), dim=0)
    rotation = quarternion.repeat((total_splats, 1))

    gaussian = GaussianModel(3)
    gaussian._xyz = positions
    gaussian._features_dc = features_dc
    gaussian._features_rest = features_rest
    gaussian._opacity = opacity
    gaussian._scaling = scaling
    gaussian._rotation = rotation

    return gaussian


if __name__ == "__main__":
    # create_test_gaussians("./output/test.ply")
    plane1 = create_plane_gaussians((0, 0, 0), (0, 1, 0), (1, 0, 0), 5, 20)
    plane2 = create_plane_gaussians((2.5, 0, 0), (0, 1, 0), (-1, 0, 0), 5, 20)
    plane1.merge_gaussians(plane2).save_ply("output/2plane_parallel.ply")