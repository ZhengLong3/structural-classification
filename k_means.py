from plyfile import PlyData, PlyElement
from gaussian-splatting.scene.gaussian_model import GaussianModel

with open("2_plane_3_edited.ply", "rb") as f:
    ply_data = PlyData.read(f)
    print(type(ply_data.elements[0].data[0]))