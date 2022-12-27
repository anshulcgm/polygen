"""Tests to ensure that the vertex model can complete a forward pass and can sample vertices"""

import pdb

import torch

from polygen.modules.vertex_model import VertexModel

torch.manual_seed(42)


def test_vertex_model_forward():
    decoder_config = {
        "hidden_size": 128,
        "fc_size": 256,
        "num_heads": 4,
        "layer_norm": True,
        "num_layers": 2,
    }

    vertex_model = VertexModel(
        decoder_config=decoder_config,
        quantization_bits=8,
        class_conditional=True,
        num_classes=10,
        max_num_input_verts=100,
        use_discrete_embeddings=True,
    )
    vertex_model_batch = {
        "vertices_flat": torch.randint(low=0, high=255, size=[4, 30]),
        "class_label": torch.randint(low=0, high=10, size=[4]),
    }
    logits = vertex_model(vertex_model_batch)


def test_vertex_model_sample():
    decoder_config = {
        "hidden_size": 128,
        "fc_size": 256,
        "num_heads": 4,
        "layer_norm": True,
        "num_layers": 2,
    }

    vertex_model = VertexModel(
        decoder_config=decoder_config,
        quantization_bits=8,
        class_conditional=True,
        num_classes=10,
        max_num_input_verts=100,
        use_discrete_embeddings=True,
    )

    context = {"class_label": torch.randint(low=0, high=10, size=[4])}
    samples = vertex_model.sample(num_samples=4, context=context)
