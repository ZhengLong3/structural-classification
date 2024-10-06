import torch

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

if __name__ == "__main__":
    create_test_gaussians("./output/test.ply")