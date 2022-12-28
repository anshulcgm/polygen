"""Tests to ensure that the face model can complete a forward pass and sample face indices"""

import pdb

import torch

from polygen.modules.face_model import FaceModel

torch.manual_seed(42)


def test_face_model_forward():
    transformer_config = {
        "hidden_size": 128,
        "fc_size": 256,
        "num_heads": 4,
        "layer_norm": True,
        "num_layers": 2,
    }
    face_model = FaceModel(
        encoder_config=transformer_config,
        decoder_config=transformer_config,
        class_conditional=True,
        num_classes=10,
    )
    class_labels = torch.randint(low=0, high=10, size=[4])
    vertices = torch.rand(size=[4, 20, 3]) - 0.5
    vertices_mask = torch.ones(size=[4, 20])
    faces = torch.randint(low=0, high=20, size=[4, 80])
    batch = {
        "faces": faces,
        "vertices": vertices,
        "vertices_mask": vertices_mask,
        "class_label": class_labels,
    }
    logits = face_model(batch)

def test_face_model_sampling():
    transformer_config = {
        "hidden_size": 128,
        "fc_size": 256,
        "num_heads": 4,
        "layer_norm": True,
        "num_layers": 2,
    }
    face_model = FaceModel(
        encoder_config=transformer_config,
        decoder_config=transformer_config,
        class_conditional=True,
        num_classes=10,
    )
    
    class_labels = torch.randint(low=0, high=10, size=[4])
    vertices = torch.rand(size=[4, 20, 3]) - 0.5
    vertices_mask = torch.ones(size=[4, 20])
    context = {"vertices": vertices, "vertices_mask": vertices_mask, "class_label": class_labels}
    samples = face_model.sample(context = context)