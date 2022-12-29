"""Tests to ensure that the data modules load .obj files correctly"""
import pdb
import random

import torch

from polygen.modules.data_modules import ShapenetDataset, PolygenDataModule, collate_vertex_model_batch, collate_face_model_batch

random.seed(10)

DATA_DIR = "/coc/scratch/aahluwalia30/shapenet/shapenetcore/ShapeNetCore"

def test_shapenet_dataset():
    dataset = ShapenetDataset(training_dir = DATA_DIR)
    dataset_len = len(dataset)
    rand_index = random.randint(0, dataset_len)
    random_object = dataset[rand_index]

def test_polygen_data_module_vertices():
    vertex_data_module = PolygenDataModule(data_dir = DATA_DIR, collate_fn = collate_vertex_model_batch, batch_size = 4)
    vertex_data_module.setup()

    train_dataloader = vertex_data_module.train_dataloader()
    val_dataloader = vertex_data_module.val_dataloader()
    test_dataloader = vertex_data_module.test_dataloader()

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))
    test_batch = next(iter(test_dataloader))

def test_polygen_data_module_faces():
    face_data_module = PolygenDataModule(data_dir = DATA_DIR, collate_fn = collate_face_model_batch, batch_size = 4)
    face_data_module.setup()

    train_dataloader = face_data_module.train_dataloader()
    val_dataloader = face_data_module.val_dataloader()
    test_dataloader = face_data_module.test_dataloader()

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))
    test_batch = next(iter(test_dataloader))
