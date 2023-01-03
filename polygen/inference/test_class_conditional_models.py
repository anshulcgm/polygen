from typing import Dict, Union
import os
import pdb

import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from pytorch3d.io import save_obj

import hydra
from hydra.utils import instantiate

from polygen.polygen_config import VertexModelConfig, FaceModelConfig
import polygen.utils.data_utils as data_utils

VERTEX_MODEL_CHECKPOINT_FILE = "/srv/share2/aahluwalia30/polygen/lightning_logs/version_491089/checkpoints/trained_vertex_model.ckpt"


def sample_from_vertex_model(
    vertex_model: pl.LightningModule, context: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Runs sampling procedure from vertex model"""
    num_samples = context["class_label"].shape[0]
    vertex_model.eval()
    with torch.no_grad():
        vertex_samples = vertex_model.sample(
            context=context, num_samples=num_samples, only_return_complete=False, max_sample_length=200
        )
    return vertex_samples


def load_vertex_model(config: VertexModelConfig) -> pl.LightningModule:
    """Loads vertex model from config object and .ckpt file"""
    model = config.vertex_model
    model.load_from_checkpoint(
        VERTEX_MODEL_CHECKPOINT_FILE,
        decoder_config=model.decoder_config,
        quantization_bits=model.quantization_bits,
        class_conditional=model.class_conditional,
        num_classes=model.num_classes,
        max_num_input_verts=model.max_num_input_verts,
        use_discrete_embeddings=model.use_discrete_embeddings,
    )
    return model


def load_config(config_name: str, vertex_config: bool) -> Union[VertexModelConfig, FaceModelConfig]:
    """Loads config object using hydra"""
    with hydra.initialize_config_module(config_module="polygen.config"):
        cfg = hydra.compose(config_name=config_name)
        if vertex_config:
            config = instantiate(cfg.VertexModelConfig)
        else:
            config = instantiate(cfg.FaceModelConfig)

    return config


def write_vertices_to_obj_files(samples: Dict[str, torch.Tensor], directory_name: str) -> None:
    """Writes generated vertices to .obj files"""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    vertices = samples["vertices"]
    num_vertices = samples["num_vertices"]
    for i in range(vertices.shape[0]):
        curr_verts = vertices[i, : num_vertices[i]]
        curr_faces = torch.zeros([0, 3]).to(torch.long)
        save_path = os.path.join(directory_name, f"{i}.obj")
        save_obj(save_path, curr_verts, curr_faces)

def plot_vertices(samples: Dict[str, torch.Tensor]) -> None:
    """Plot generated vertices using matplotlib"""
    vertices = samples["vertices"]
    num_vertices = samples["num_vertices"]
    mesh_list = []
    for i in range(vertices.shape[0]):
        curr_verts = vertices[i, :num_vertices[i]].numpy()
        mesh_list.append({'vertices': curr_verts})
    
    data_utils.plot_meshes(mesh_list, ax_lims = 0.5)
    plt.savefig('generated_meshes.png')

def test_vertex_model(config_name: str) -> None:
    """Intitializes vertex model and samples from it"""
    vertex_model_config = load_config(config_name=config_name, vertex_config = True)
    model = load_vertex_model(vertex_model_config)
    context = {"class_label": torch.Tensor([0, 1, 2, 3])}
    samples = sample_from_vertex_model(model, context)
    plot_vertices(samples)

if __name__ == "__main__":
    test_vertex_model(config_name = "vertex_model_config_1231.yaml")
