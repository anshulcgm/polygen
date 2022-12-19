"""Utils for manipulating obj data"""
import os
import six
from six.moves import range
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .truncated_normal import TruncatedNormal


def random_shift(vertices: torch.Tensor, shift_factor: float = 0.25) -> torch.Tensor:
    """Randomly shift vertices in a cube according to some shift factor

    Args:
        vertices: tensor of shape (num_vertices, 3) representing current vertices
        shift_factor: float representing how much vertices should be shifted

    Returns:
        vertices: Shifted vertices
    """

    max_positive_shift = (255 - torch.max(vertices, axis=0)[0]).to(torch.float32)
    positive_condition_tensor = max_positive_shift > 1e-9
    max_positive_shift = torch.where(
        positive_condition_tensor, max_positive_shift, torch.Tensor([1e-9, 1e-9, 1e-9])
    )

    max_negative_shift = torch.min(vertices, axis=0)[0].to(torch.float32)
    negative_condition_tensor = max_negative_shift > 1e-9
    max_negative_shift = torch.where(
        negative_condition_tensor, max_negative_shift, torch.Tensor([1e-9, 1e-9, 1e-9])
    )
    normal_dist = TruncatedNormal(
        loc=torch.zeros((1, 3)),
        scale=shift_factor * 255,
        a=-max_negative_shift,
        b=max_positive_shift,
    )
    shift = normal_dist.sample().to(torch.int32)
    vertices += shift
    return vertices


def read_obj(obj_path: str) -> Tuple[np.ndarray, List]:
    """Utils method to read .obj file

    Args:
        obj_path: Path of .obj file
    Returns:
        flat_vertices_list: flattened vertex coordinates
        flat_triangles: vertex indices representing connectivity
    """
    vertex_list = []
    flat_vertices_list = []
    flat_vertices_indices = {}
    flat_triangles = []

    with open(obj_path) as obj_file:
        for line in obj_file:
            tokens = line.split()
            if not tokens:
                continue
            line_type = tokens[0]
            # We skip lines not starting with v or f.
            if line_type == "v":
                vertex_list.append([float(x) for x in tokens[1:]])
            elif line_type == "f":
                triangle = []
                for i in range(len(tokens) - 1):
                    vertex_name = tokens[i + 1]
                    if vertex_name in flat_vertices_indices:
                        triangle.append(flat_vertices_indices[vertex_name])
                        continue
                    flat_vertex = []
                    for index in six.ensure_str(vertex_name).split("/"):
                        if not index:
                            continue
                        # obj triangle indices are 1 indexed, so subtract 1 here.
                        flat_vertex += vertex_list[int(index) - 1]
                    flat_vertex_index = len(flat_vertices_list)
                    flat_vertices_list.append(flat_vertex)
                    flat_vertices_indices[vertex_name] = flat_vertex_index
                    triangle.append(flat_vertex_index)
                flat_triangles.append(triangle)

    return np.array(flat_vertices_list, dtype=np.float32), flat_triangles

def write_obj(vertices: np.ndarray, faces: List, file_path: str, transpose: bool = True, scale: float = 1.) -> None:
    """Writes vertices and faces to .obj file to represent 3D object

    Args:
        vertices: array of shape (num_vertices, 3) representing vertex indices
        faces: List of vertex indices representing vertex connectivity
        file_path: Where to save .obj file
        transpose: boolean representing whether to change traditional order of (x, y, z)
        scale: Factor by which to scale vertices
    """
    if transpose:
        vertices = vertices[:, [1, 2, 0]]
    vertices *= scale
    if faces is not None:
        if min(min(faces)) == 0:
            f_add = 1
        else:
            f_add = 0
    with open(file_path, 'w') as f:
        for v in vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for face in faces:
            line = 'f'
            for i in face:
                line += ' {}'.format(i + f_add)
            line += '\n'
            f.write(line)

