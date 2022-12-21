from typing import List, Dict, Optional
import random
import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch3d.io import load_obj
import pytorch_lightning as pl

import polygen.utils.data_utils as data_utils


def collate_vertex_model_batch(ds: List[Dict[str, torch.Tensor]], apply_random_shift: bool = True) -> Dict[str, torch.Tensor]:
    """Given vertex sequences of different lengths, combine them together

    Args:
        ds: List of dictionaries with vertices
    
    Returns:
        vertex_model_batch: One dictionary that contains collated vertices and masks and class labels
    """

    vertex_model_batch = {}
    num_vertices_list = [shape_dict["vertices"].shape[0] for shape_dict in ds]
    max_vertices = max(num_vertices_list)
    num_elements = len(ds)
    vertices_flat = torch.zeros([num_elements, max_vertices * 3 + 1])
    class_labels = torch.zeros([num_elements])
    vertices_flat_mask = torch.zeros_like(vertices_flat)
    for i, element in enumerate(ds):
        vertices = element["vertices"]
        if apply_random_shift:
            vertices = random_shift(vertices)
        initial_vertex_size = vertices.shape[0]
        padding_size = max_vertices - initial_vertex_size
        vertices_permuted = torch.stack(
            [vertices[..., 2], vertices[..., 1], vertices[..., 0]], dim=-1
        )
        curr_vertices_flat = vertices_permuted.reshape([-1])
        vertices_flat[i] = F.pad(curr_vertices_flat + 1, [0, padding_size * 3 + 1])[
            None
        ]
        class_labels[i] = torch.Tensor([element["class_label"]])
        vertices_flat_mask[i] = torch.zeros_like(vertices_flat[i], dtype=torch.float32)
        vertices_flat_mask[i, : initial_vertex_size * 3 + 1] = 1
    vertex_model_batch["vertices_flat"] = vertices_flat
    vertex_model_batch["class_label"] = class_labels
    vertex_model_batch["vertices_flat_mask"] = vertices_flat_mask
    return vertex_model_batch


def collate_face_model_batch(
    ds, apply_random_shift=True, shuffle_vertices=True, quantization_bits=8
):
    face_model_batch = {}
    num_vertices_list = [shape_dict["vertices"].shape[0] for shape_dict in ds]
    max_vertices = max(num_vertices_list)
    num_faces_list = [shape_dict["faces"].shape[0] for shape_dict in ds]
    max_faces = max(num_faces_list)
    num_elements = len(ds)

    shuffled_faces = torch.zeros([num_elements, max_faces])
    face_vertices = torch.zeros([num_elements, max_vertices, 3])
    face_vertices_mask = torch.zeros([num_elements, max_vertices])
    faces_mask = torch.zeros_like(shuffled_faces)

    for i, element in enumerate(ds):
        vertices = element["vertices"]

        if apply_random_shift:
            vertices = random_shift(vertices)
        num_vertices = vertices.shape[0]
        if shuffle_vertices:
            permutation = torch.randperm(num_vertices)
            vertices = vertices[permutation]
            vertices = vertices.unsqueeze(0)
            face_permutation = torch.cat(
                [
                    torch.Tensor([0, 1]).to(torch.int32),
                    torch.argsort(permutation).to(torch.int32) + 2,
                ],
                axis=0,
            )
            curr_faces = face_permutation[element["faces"]][None]
        else:
            curr_faces = faces

        vertex_padding_size = max_vertices - num_vertices
        initial_faces_size = curr_faces.shape[1]
        face_padding_size = max_faces - initial_faces_size
        shuffled_faces[i] = F.pad(curr_faces, [0, face_padding_size, 0, 0])
        curr_verts = helper_methods.dequantize_verts(
            vertices, quantization_bits
        )
        face_vertices[i] = F.pad(curr_verts, [0, 0, 0, vertex_padding_size])
        face_vertices_mask[i] = torch.zeros_like(
            face_vertices[i][..., 0], dtype=torch.float32
        )
        face_vertices_mask[i, :num_vertices] = 1
        faces_mask[i] = torch.zeros_like(shuffled_faces[i], dtype=torch.float32)
        faces_mask[i, : initial_faces_size + 1] = 1
    face_model_batch["faces"] = shuffled_faces
    face_model_batch["vertices"] = face_vertices
    face_model_batch["vertices_mask"] = face_vertices_mask
    face_model_batch["faces_mask"] = faces_mask
    return face_model_batch
 
class ShapenetDataset(Dataset):
    """Dataset object for class-conditioned training on shapenet"""
    def __init__(self, training_dir: str) -> None:
        """
        Args:
            training_dir: Shapenet directory path
        """
        self.training_dir = training_dir
        self.all_files = glob.glob(f"{self.training_dir}/*/*/models/model_normalized.obj")
        self.label_dict = {}
        for i, class_label in enumerate(os.listdir(self.training_dir)):
            self.label_dict[class_label] = i
    
    def __len__(self) -> int:
        """Returns number of 3D objects in dataset"""
        return len(self.all_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return vertices, faces and class label for object at current idx

        Args:
            idx: which index of the all_files list we want to load

        Returns:
            mesh_dict: Dictionary containing vertices, faces and class label
        """

        mesh_file = self.all_files[idx]
        verts, faces, _ = load_obj(mesh_file)
        faces = faces.verts_idx
        vertices = verts[:, [2, 0, 1]] #Switch from (x, y, z) to (z, x, y)
        vertices = vertices.numpy()
        faces = faces.numpy()
        vertices = data_utils.center_vertices(vertices)
        vertices = data_utils.normalize_vertices_scale(vertices)
        vertices, faces, _ = data_utils.quantize_process_mesh(vertices, faces)
        faces = data_utils.flatten_faces(faces)
        vertices = torch.from_numpy(vertices)
        faces = torch.from_numpy(faces)

        class_label = self.label_dict[mesh_file.split("/")[-4]]
        mesh_dict = {"vertices": vertices, "faces": faces, "class_label": class_label}
        return mesh_dict

