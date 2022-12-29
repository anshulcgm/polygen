import glob
import os
import pdb
import random
from typing import List, Dict, Optional, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch3d.io import load_obj
import pytorch_lightning as pl

import polygen.utils.data_utils as data_utils


def collate_vertex_model_batch(
    ds: List[Dict[str, torch.Tensor]], apply_random_shift: bool = True
) -> Dict[str, torch.Tensor]:
    """Applying padding to different length vertex sequences so we can batch them
    Args:
        ds: List of dictionaries where each dictionary has information about a 3D object
        apply_random_shift: If we want to add noise to each vertex
    Returns
        vertex_model_batch: A single dictionary which represents the whole batch
    """
    vertex_model_batch = {}
    num_vertices_list = [shape_dict["vertices"].shape[0] for shape_dict in ds]
    max_vertices = max(num_vertices_list)
    num_elements = len(ds)
    vertices_flat = torch.zeros([num_elements, max_vertices * 3 + 1], dtype=torch.int32)
    class_labels = torch.zeros([num_elements], dtype=torch.int32)
    vertices_flat_mask = torch.zeros_like(vertices_flat, dtype=torch.int32)
    for i, element in enumerate(ds):
        vertices = element["vertices"]
        if apply_random_shift:
            vertices = data_utils.random_shift(vertices)
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
    ds: List[Dict[str, torch.Tensor]],
    apply_random_shift: bool = True,
    shuffle_vertices: bool = True,
    quantization_bits: int = 8,
) -> Dict[str, torch.Tensor]:
    """Applies padding to different length face sequences so we can batch them
    Args:
        ds: List of dictionaries with each dictionary containing info about a specific 3D object
        apply_random_shift: If we want to add noise to each vertex
        shuffle_vertices: If we want to shuffle the vertices to help generalization
        quantization_bits: How many bits we are using to convert the vertices to integers
    Returns:
        face_model_batch: A single dictionary which represents the whole face model batch
    """
    face_model_batch = {}
    num_vertices_list = [shape_dict["vertices"].shape[0] for shape_dict in ds]
    max_vertices = max(num_vertices_list)
    num_faces_list = [shape_dict["faces"].shape[0] for shape_dict in ds]
    max_faces = max(num_faces_list)
    num_elements = len(ds)

    shuffled_faces = torch.zeros([num_elements, max_faces], dtype=torch.int32)
    face_vertices = torch.zeros([num_elements, max_vertices, 3])
    face_vertices_mask = torch.zeros([num_elements, max_vertices], dtype=torch.int32)
    faces_mask = torch.zeros_like(shuffled_faces, dtype=torch.int32)

    for i, element in enumerate(ds):
        vertices = element["vertices"]
        num_vertices = vertices.shape[0]
        if apply_random_shift:
            vertices = data_utils.random_shift(vertices)

        if shuffle_vertices:
            permutation = torch.randperm(num_vertices)
            vertices = vertices[permutation]
            vertices = vertices.unsqueeze(0)
            face_permutation = torch.cat(
                [
                    torch.Tensor([0, 1]).to(torch.int32),
                    torch.argsort(permutation).to(torch.int32) + 2,
                ],
                dim=0,
            )
            curr_faces = face_permutation[element["faces"].to(torch.int64)][None]
        else:
            curr_faces = faces

        vertex_padding_size = max_vertices - num_vertices
        initial_faces_size = curr_faces.shape[1]
        face_padding_size = max_faces - initial_faces_size
        shuffled_faces[i] = F.pad(curr_faces, [0, face_padding_size, 0, 0])
        curr_verts = data_utils.dequantize_verts(vertices, quantization_bits)
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
    def __init__(
        self,
        training_dir: str,
        default_shapenet: bool = True,
        all_files: Optional[List[str]] = None,
        label_dict: Dict[str, int] = None,
    ) -> None:
        """
        Args:
            training_dir: Root folder of shapenet dataset
        """
        self.training_dir = training_dir
        self.default_shapenet = default_shapenet
        if default_shapenet:
            self.all_files = glob.glob(
                f"{self.training_dir}/*/*/models/model_normalized.obj"
            )
            self.label_dict = {}
            for i, class_label in enumerate(os.listdir(training_dir)):
                self.label_dict[class_label] = i
        else:
            self.all_files = all_files
            self.label_dict = label_dict

    def __len__(self) -> int:
        """Returns number of 3D objects"""
        return len(self.all_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns processed vertices, faces and class label of a mesh
        Args:
            idx: Which 3D object we're retrieving
        Returns:
            mesh_dict: Dictionary containing vertices, faces and class label
        """
        mesh_file = self.all_files[idx]
        vertices, faces, _ = load_obj(mesh_file)
        faces = faces.verts_idx
        vertices = vertices[:, [2, 0, 1]]
        vertices = data_utils.center_vertices(vertices)
        vertices = data_utils.normalize_vertices_scale(vertices)
        vertices, faces, _ = data_utils.quantize_process_mesh(vertices, faces)
        faces = data_utils.flatten_faces(faces)
        vertices = vertices.to(torch.int32)
        faces = faces.to(torch.int32)
        if self.default_shapenet:
            class_label = self.label_dict[mesh_file.split("/")[-4]]
        else:
            class_label = self.label_dict[mesh_file]
        mesh_dict = {"vertices": vertices, "faces": faces, "class_label": class_label}
        return mesh_dict


class PolygenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        collate_fn: Callable,
        batch_size: int,
        training_split: float = 0.925,
        val_split: float = 0.025,
        default_shapenet: bool = True,
        all_files: Optional[List[str]] = None,
        label_dict: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Args:
            data_dir: Root directory for shapenet dataset
            batch_size: How many 3D objects in one batch
            training_split: What proportion of data to use for training the model
            val_split: What proportion of data to use for validation
            default_shapenet: Whether or not we are using the default shapenet data structure
            all_files: List of all .obj files (needs to be provided if default_shapnet = false)
            label_dict: Mapping of .obj file to class label (needs to be provided if default_shapnet = false)
        """
        super().__init__()

        assert (training_split + val_split) <= 1.0

        self.data_dir = data_dir
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.shapenet_dataset = ShapenetDataset(self.data_dir, default_shapenet = default_shapenet, all_files = all_files, label_dict = label_dict)
        self.training_split = training_split
        self.val_split = val_split

    def setup(self, stage: Optional = None) -> None:
        """Pytorch Lightning Data Module setup method"""
        num_files = len(self.shapenet_dataset)
        train_set_length = int(num_files * self.training_split)
        val_set_length = int(num_files * self.val_split)
        test_set_length = num_files - train_set_length - val_set_length
        self.train_set, self.val_set, self.test_set = random_split(
            self.shapenet_dataset, [train_set_length, val_set_length, test_set_length]
        )

    def train_dataloader(self) -> DataLoader:
        """
        Returns:
            train_dataloader: Dataloader used to load training batches
        """
        return DataLoader(
            self.train_set,
            self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=16,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns:
            val_dataloader: Dataloader used to load validation batches
        """
        return DataLoader(
            self.val_set,
            self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns:
            test_dataloader: Dataloader used to load test batches
        """
        return DataLoader(
            self.test_set,
            self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=8,
            persistent_workers=True,
        )
