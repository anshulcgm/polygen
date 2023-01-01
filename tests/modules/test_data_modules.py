"""Tests to ensure that the data modules load .obj files correctly, batch files correctly and can pass files into models"""
import pdb
import random

import torch

from polygen.modules.data_modules import ShapenetDataset, PolygenDataModule, CollateMethod
from polygen.modules.face_model import FaceModel
from polygen.modules.vertex_model import VertexModel

random.seed(10)

DATA_DIR = "/coc/scratch/aahluwalia30/shapenet/shapenetcore/ShapeNetCore"


def test_shapenet_dataset():
    dataset = ShapenetDataset(training_dir=DATA_DIR)
    dataset_len = len(dataset)
    rand_index = random.randint(0, dataset_len)
    random_object = dataset[rand_index]


def test_polygen_data_module_vertices():
    vertex_data_module = PolygenDataModule(data_dir=DATA_DIR, collate_method=CollateMethod.VERTICES, batch_size=4)
    vertex_data_module.setup()

    train_dataloader = vertex_data_module.train_dataloader()
    val_dataloader = vertex_data_module.val_dataloader()
    test_dataloader = vertex_data_module.test_dataloader()

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))
    test_batch = next(iter(test_dataloader))

    transformer_config = {
        "hidden_size": 128,
        "fc_size": 256,
        "num_heads": 4,
        "layer_norm": True,
        "num_layers": 2,
    }

    vertex_model = VertexModel(
        decoder_config=transformer_config,
        quantization_bits=8,
        class_conditional=True,
        num_classes=55,
        use_discrete_embeddings=True,
    )

    logits = vertex_model(train_batch)
    samples = vertex_model.sample(num_samples=4, context=train_batch)


def test_polygen_data_module_faces():
    face_data_module = PolygenDataModule(data_dir=DATA_DIR, collate_method=CollateMethod.FACES, batch_size=4)
    face_data_module.setup()

    train_dataloader = face_data_module.train_dataloader()
    val_dataloader = face_data_module.val_dataloader()
    test_dataloader = face_data_module.test_dataloader()

    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))
    test_batch = next(iter(test_dataloader))

    transformer_config = {
        "hidden_size": 128,
        "fc_size": 256,
        "num_heads": 4,
        "layer_norm": True,
        "num_layers": 2,
    }

    face_model = FaceModel(
        encoder_config=transformer_config, decoder_config=transformer_config, class_conditional=False, num_classes=55,
    )

    logits = face_model(train_batch)
    samples = face_model.sample(context=train_batch)
