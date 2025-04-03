from __future__ import annotations

import json
import math

import networkx as nx
import numpy as np
import open3d as o3d
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from gaussian_model import GaussianModel

def get_random_color() -> np.ndarray[np.float64[3]]:
    return np.random.rand(3)


class PlaneNode:
    def __init__(self, center: list, extent: list, rotmat: list[list], factor: float = 1.00) -> None:
        self.center = np.array(center, np.float64)
        self.extent = np.array(extent, np.float64)
        self.extent *= factor
        self.rotmat = np.array(rotmat, np.float64)
        self.vec1 = np.array(rotmat[:, 0], np.float64)
        self.vec2 = np.array(rotmat[:, 1], np.float64)
        self.normal = np.array(rotmat[:, 2], np.float64)
        self.color = get_random_color()

    def __str__(self) -> str:
        return f"[Center: {str(self.center)}, Extent: {str(self.extent)}, Rotation Matrix: {str(self.rotmat)}]"

    @classmethod
    def create_from_obox(cls, obox: o3d.geometry.OrientedBoundingBox, factor: float = 1.00) -> PlaneNode:
        """
        Create a PlaneNode object using an Open3D OrientedBoundingBox object.

        Parameters:
            obox: Open3D OrientedBoundingBox object describing a rectangular plane.
            factor: Float to scale extent by, since planes might not overlap otherwise.

        Returns:
            A PlaneNode describing the rectangular plane.
        """
        return cls(obox.center, obox.extent, obox.R, factor)

    @classmethod
    def create_from_vectors(cls, center: tuple[float, float, float], side1: tuple[float, float, float], side2: tuple[float, float, float], factor: float = 1.0):
        """
        Creates a PlaneNode with a center and two vectors describing the direction of the two sides. Gram-Schmidt orthogonalisation is done in the order of side1 then side2 to obtain orthogonal vectors.

        Parameters:
            center: Position of the center of the rectangle.
            side1: 3D vector in the form of a tuple describing the length and direction of the first side.
            side2: 3D vector in the form of a tuple describing the length and direction of the second side.
            factor: Float to scale extent by, since planes might not overlap otherwise.

        Returns:
            A PlaneNode described by the two side vectors and center.
        """
        extent = np.zeros(3, dtype=np.float64)
        vector1 = np.array(side1)
        vector2 = np.array(side2)
        vector2 = vector2 - np.dot(vector1, vector2) / np.dot(vector1, vector1) * vector1
        extent[0] = np.linalg.norm(vector1, 2)
        extent[1] = np.linalg.norm(vector2, 2)
        extent[2] = 1
        center = np.array(center)
        rotmat = np.column_stack((vector1 / extent[0], vector2 / extent[1], np.cross(vector1 / extent[0], vector2 / extent[1])))
        return cls(center, extent, rotmat, factor)

    @classmethod
    def create_from_corners(cls, corner1: tuple[float, float, float], corner2: tuple[float, float, float], corner3: tuple[float, float, float], corner4: tuple[float, float, float]) -> PlaneNode:
        """
        Creates a PlaneNode with the four corners of the rectangle it should represent.
        The center will be the average of the four corners' coordinates and the edges will be the average of the two parallel sides.
        The corners should be in clockwise/anticlockwise order.

        Parameters:
            corner1: The first corner as a tuple of three floats.
            corner2: The second corner as a tuple of three floats. It should be adjacent to the first corner.
            corner3: The third corner as a tuple of three floats. It should be adjacent to the second corner.
            corner4: The fourth corner as a tuple of three floats. It should be adjacent to the first and third corner.

        Returns:
            A PlaneNode representing the rectangle described by the corners.
        """
        c1 = np.array(corner1, dtype=np.float64)
        c2 = np.array(corner2, dtype=np.float64)
        c3 = np.array(corner3, dtype=np.float64)
        c4 = np.array(corner4, dtype=np.float64)
        center = (c1 + c2 + c3 + c4) / 4.0
        vector1 = (c2 - c1 + c3 - c4) / 2
        vector2 = (c4 - c1 + c3 - c2) / 2
        return cls.create_from_vectors(center, vector1, vector2)


    def get_vectors(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """
        Returns the center, and two side vectors which describes the rectangle.

        Returns:
            Tuple containing the center (length 3 tuple), and the two side vectors (also length 3 tuples).
        """
        return (tuple(self.center), tuple(self.rotmat[:, 0] * self.extent[0]), tuple(self.rotmat[:, 1] * self.extent[1]))

    def get_area(self) -> float:
        """
        Returns the area of the rectangular plane that the node represents.
        The third value of the extent is the "error" of the plane as it is in the direction of the normal vector.

        Returns:
            The area of the plane.
        """
        return (self.extent[0] * self.extent[1]).item()

    def get_normalised_area(self, max_area: float) -> float:
        """
        Returns the normalised area, after dividing the area by the max_area.

        Parameters:
            max_area: The maximum area of a node in a StructureGraph

        Returns:
            The normalised area.
        """
        return self.get_area() / max_area

    def check_overlap(self, other: PlaneNode) -> bool:
        """
        Checks if this plane overlaps with another plane.

        Parameters:
            other: The other PlaneNode to check against.

        Returns:
            Boolean for whether this plane overlaps with another plane.
        """

        """
        We formulate the problem as a linear program. Let the center of the two rectangular planes be c_1 and c_2 respectively.
        Let v_11, v_12, v_21, v_22 denote the vectors along the two sides of the rectangles of both planes respectively.
        Let e_11, e_12, e_21, e_22 be the extents in both directions for both planes.
        Finally, let x_11, x_12, x_21, x_22 be variables denoting the scalar multiples of v_11, v_12, v_21, v_22 respectively,
        describing a point on the plane of the rectangles. Then we have the following constraints:

        x_ab <= e_ab / 2, since we want the point to be within the rectangles.
        -x_ab <= e_ab / 2, for the negative case
        c_1 + x_11 * v_11 + x_12 * v_12 = c_2 + x_21 * v_21 + x_22 * v_22,
        since we want a common point within both rectangles to determine overlap.

        If such x_ab exists, then both rectangles overlap. Otherwise, the rectangles do not overlap.
        """

        objective_function = [1, 1, 1, 1]
        A_upp = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
        b_upp = [self.extent[0] / 2, self.extent[1] / 2, other.extent[0] / 2, other.extent[1] / 2, self.extent[0] / 2, self.extent[1] / 2, other.extent[0] / 2, other.extent[1] / 2]
        A_eq = [[self.vec1[0], self.vec2[0], -other.vec1[0], -other.vec2[0]],
                [self.vec1[1], self.vec2[1], -other.vec1[1], -other.vec2[1]],
                [self.vec1[2], self.vec2[2], -other.vec1[2], -other.vec2[2]]]
        b_eq = [other.center[0] - self.center[0], other.center[1] - self.center[1], other.center[2] - self.center[2]]
        result = scipy.optimize.linprog(objective_function, A_upp, b_upp, A_eq, b_eq, (None, None))
        return result.success

    def get_obox(self) -> o3d.geometry.OrientedBoundingBox:
        """
        Returns the plane as an Open3D OrientedBoundingBox.

        Returns:
            OrientedBoundingBox of the rectangular plane.
        """
        return o3d.geometry.OrientedBoundingBox(center = np.expand_dims(self.center, -1), R = self.rotmat, extent = np.expand_dims(self.extent, -1))

    def get_corners(self) -> np.ndarray[np.float64[4, 3]]:
        """
        Returns the coordinates of the rectangle's corners in a 3 by 4 array, where each column is the (x, y, z) coordinates of a corner.

        Returns:
            Numpy ndarray of corners of rectangle.
        """
        scalars = [(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)]
        corners = []
        for scalar in scalars:
            corners.append(self.center + self.extent[0] * self.vec1 * scalar[0] + self.extent[1] * self.vec2 * scalar[1])
        return np.vstack(corners)

    def get_angle_to(self, other: PlaneNode) -> float:
        """
        Returns the angle between this plane and another plane (assuming planes are infinite).
        The angle returned will be the angle from the center of a plane to the intersection, to the center of the other plane.
        This attempts to be the intuitive angle between two connected real-life planes.

        Parameters:
            other: The other PlaneNode to compute the angle to.

        Returns:
            Angle between the two planes, between 0 and pi radians.
        """
        SCALING = 100
        normal1 = self.normal
        # I think the angle is correct if we choose 1 normal to be towards the other center, and one away.
        if np.linalg.norm(self.center + normal1 / SCALING - other.center) > np.linalg.norm(self.center - normal1 / SCALING - other.center):
            # normal of plane 1 to be towards center of plane 2.
            normal1 = - normal1
        normal2 = other.normal
        if np.linalg.norm(other.center + normal2 / SCALING - self.center) < np.linalg.norm(other.center - normal2 / SCALING - self.center):
            # normal of plane 2 to be away from the center of plane 1.
            normal2 = - normal2
        return math.acos(np.dot(normal1, normal2))

    def get_overlap_vector(self, other: PlaneNode) -> np.ndarray:
        """
        Get the direction of the line of intersection between the PlaneNode itself and another PlaneNode.
        This assumes the planes are infinite, so most likely there will be an intersection.

        Parameters:
            other: The other PlaneNode to compute the overlap vector.

        Returns:
            The direction vector of the intersecting line.
        """
        return np.cross(self.normal, other.normal)


class StructureGraph:
    def __init__(self) -> None:
        """
        Initialises the class with an empty node and edge list.
        """
        self.graph: nx.Graph = nx.Graph()
        self._max_size = 0

    def __str__(self) -> str:
        output = "Nodes: "
        for node in self.nodes:
            output += str(node) + " "
        output += "\nEdges: "
        for edge in self.edges:
            output += str(edge) + " "
        return output

    @classmethod
    def create_from_gaussians(cls, gaussian: GaussianModel, device: str = "CPU:0", factor: float = 1.1) -> StructureGraph:
        """
        Extracts the planes in a Gaussian Model using Open3D and creates a structure graph based on the planes.

        Parameters:
            gaussian: Gaussian Model containing the object to be analysed.
            device: The device to use to extract the planes. Default is "CPU:0".
            factor: Float to scale extent by, since planes might not overlap otherwise.

        Returns:
            StructureGraph for the planes detected in the GaussianModel.
        """
        structure_graph = cls()

        o3d_device = o3d.core.Device(device)
        dtype = o3d.core.float32

        # Obtaining planes
        pcd = o3d.t.geometry.PointCloud(o3d_device)
        pcd.point.positions = o3d.core.Tensor(gaussian._xyz.numpy(), dtype, o3d_device)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(10)
        pcd = pcd.to_legacy()
        oboxes = pcd.detect_planar_patches(
            normal_variance_threshold_deg=30,
            coplanarity_deg=70,
            outlier_ratio=0.4,
            min_plane_edge_length=0,
            min_num_points=0,
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        for obox in oboxes:
            structure_graph.add_node(PlaneNode.create_from_obox(obox, factor=factor))

        structure_graph.populate_edge_list()

        return structure_graph

    @classmethod
    def create_from_node_list(cls, node_list: list[PlaneNode]) -> StructureGraph:
        """
        Create a structure graph object from a list of Rectangles.

        Parameters:
            node_list: A list of PlaneNodes which make up the structure.

        Returns:
            A StructureGraph object described by the list of PlaneNodes.
        """
        structure_graph = cls()
        for node in node_list:
            structure_graph.add_node(node)
        structure_graph.populate_edge_list()
        return structure_graph
    
    @classmethod
    def create_from_vector_list(cls, vector_list: list[list[float]]):
        return StructureGraph.create_from_node_list(map(lambda x: PlaneNode.create_from_vectors(*x), vector_list))

    @classmethod
    def create_from_json(cls, path: str) -> StructureGraph:
        """
        Create a structure graph object from json format.

        Parameters:
            path: String for the path to the json file.

        Returns:
            A StructureGraph object described by the json file.
        """
        with open(path, "r") as f:
            rectangle_list = json.load(f)
        return cls.create_from_node_list(map(lambda x: PlaneNode.create_from_vectors(*x) ,rectangle_list))

    def populate_edge_list(self) -> None:
        node_list: list[PlaneNode] = list(self.graph.nodes)
        for i in range(len(node_list) - 1):
            for j in range(i + 1, len(node_list)):
                if node_list[i].check_overlap(node_list[j]):
                    self.graph.add_edge(node_list[i], node_list[j])

    def add_node(self, node: PlaneNode) -> None:
        """
        Add a PlaneNode to the graph.

        Parameters:
            node: PlaneNode to add.
        """
        self.graph.add_node(node)
        self._max_size = max(self._max_size, node.get_area())

    def filter_by_size(self, threshold: float) -> None:
        """
        Remove nodes below a certain weight value.

        Parameters:
            threshold: Largest weight value required to not be removed.
        """
        max_area = self.get_max_node_area()
        node_list: list[PlaneNode] = list(self.graph.nodes)
        for node in node_list:
            if node.get_normalised_area(max_area) < threshold:
                self.graph.remove_node(node)


    def to_json(self, path: str) -> None:
        """
        Saves the structure to json format.
        """
        rectangle_list = []
        node: PlaneNode
        for node in self.graph.nodes:
            rectangle_list.append(node.get_vectors())
        with open(path, "w") as f:
            json.dump(rectangle_list, f)

    def visualise_graph(self) -> None:
        """
        Visualise the graph as a matplotlib plot.
        """
        plt.figure()
        labels = {}
        edge_labels = {}
        sizes = []
        colors = []
        node: PlaneNode
        for node in self.graph.nodes:
            labels[node] = round(node.get_normalised_area(self._max_size), 2)
            sizes.append(node.get_area() * 300)
            colors.append(node.color)
        edge: tuple[PlaneNode, PlaneNode]
        for edge in self.graph.edges:
            edge_labels[edge] = round(edge[0].get_angle_to(edge[1]), 2)
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, labels=labels, with_labels=True, node_color=colors)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

    def visualise_planes_o3d(self) -> None:
        """
        Visualise the planes using Open3D.
        """
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        node: PlaneNode
        for node in self.graph.nodes:
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(node.get_obox(), scale=[1, 1, 0.0001])
            mesh.paint_uniform_color(node.color)
            viewer.add_geometry(mesh)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        viewer.run()
        viewer.destroy_window()

    def visualise_planes(self) -> None:
        """
        Visualise the planes using Matplotlib.
        """
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_axis_off()
        faces = []
        colours = []
        for node in self.graph.nodes:
            faces.append(node.get_corners())
            colours.append(node.color)
        poly = Poly3DCollection(faces, zsort="min")
        poly.set_facecolor(colours)
        ax.add_collection3d(poly)
        ax.set_aspect("equal")

    def get_max_node_area(self) -> float:
        """
        Gets the maximum area of the nodes in this StructureGraph.

        Returns:
            Maximum area of the nodes in this StructureGraph.
        """
        node: PlaneNode
        max_area = 0
        for node in self.graph.nodes:
            max_area = max(max_area, node.get_area())
        return max_area

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Gets the adjacency matrix which represents the graph. The matrix is a symmetric matrix with the diagonals representing the weights of the nodes and the other entries representing the weights of the edge connecting two nodes if such an edge exist, and -1 otherwise.

        Returns:
            An ndarray for the adjacency matrix.
        """
        node_dict: dict[PlaneNode, int] = {}
        current_id = 0
        node: PlaneNode
        max_area = 0
        for node in self.graph.nodes:
            node_dict[node] = current_id
            max_area = max(max_area, node.get_area())
            current_id += 1
        output = np.ones((len(node_dict), len(node_dict)))
        output = -output
        for node in node_dict:
            output[node_dict[node], node_dict[node]] = node.get_normalised_area(max_area)
        edge: tuple[PlaneNode, PlaneNode]
        for edge in self.graph.edges:
            node1_id = node_dict[edge[0]]
            node2_id = node_dict[edge[1]]
            angle = edge[0].get_angle_to(edge[1])
            output[node1_id, node2_id] = angle
            output[node2_id, node1_id] = angle
        return output

    def get_simple_graph(self) -> nx.Graph:
        """
        This returns a graph which preserves the topology and weights of the graph, but leaves out the other details of the planes.
        """
        new_graph = nx.Graph()
        max_area = self.get_max_node_area()
        node_dict = {}
        current_index = 0
        node: PlaneNode
        for node in self.graph.nodes:
            node_dict[node] = current_index
            new_graph.add_node(current_index, area=node.get_normalised_area(max_area))
            current_index += 1
        edge: tuple[PlaneNode, PlaneNode]
        for edge in self.graph.edges:
            new_graph.add_edge(node_dict[edge[0]], node_dict[edge[1]], angle=edge[0].get_angle_to(edge[1]))
        return new_graph


if __name__ == "__main__":
    def load_from_gaussian_model_example():
        # loading from gaussian model and saving to file.
        gaussian = GaussianModel(3)
        gaussian.load_ply("./data/full_tetrahedron.ply")
        graph = StructureGraph.create_from_gaussians(gaussian, factor=1.1)
        graph.filter_by_size(0.3)
        graph.to_json("output/structure.json")
        graph.visualise_graph()
        graph.visualise_planes()
        plt.show()

    def load_from_structure_example():
        # loading from structures
        structures = {}
        with open("data/structures.json", "r") as f:
            structure_list = json.load(f)
        for structure in structure_list:
            structures[structure["name"]] = structure["rects"]
        graph = StructureGraph.create_from_node_list(map(lambda x: PlaneNode.create_from_vectors(*x), structures["cleaned_book"]))
        graph.visualise_graph()
        graph.visualise_planes()
        plt.show()

    # graph.visualise_planes_o3d()
    load_from_gaussian_model_example()
    # load_from_structure_example()