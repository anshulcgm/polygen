"""Tests to ensure that the vertex model can complete a forward pass and can sample a vertex"""

import pdb

import torch


from polygen.modules.vertex_model import VertexModel

torch.random.seed(42)


def test_vertex_model():
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

    vertex_model_batch = {"vertices_flat": torch.randint(low=0, high=255, size=(4, 30))}
    logits = vertex_model(vertex_model_batch)
