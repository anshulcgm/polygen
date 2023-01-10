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

VERTEX_MODEL_CHECKPOINT_FILE = (
    "/srv/share2/aahluwalia30/polygen/lightning_logs/version_491089/checkpoints/trained_vertex_model.ckpt"
)

FACE_MODEL_CHECKPOINT_FILE = (
    "/srv/share2/aahluwalia30/polygen/lightning_logs/version_494214/checkpoints/trained_face_model.ckpt"
)


def sample_from_vertex_model(
    vertex_model: pl.LightningModule, context: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Runs vertex model sampling procedure

    Args:
        vertex_model: Lightning module with trained weights
        context: Dictionary that contains class labels

    Returns
        samples: Sampled vertices along with masks and other indicator tensors
    """
    num_samples = context["class_label"].shape[0]
    vertex_model.eval()
    with torch.no_grad():
        vertex_samples = vertex_model.sample(
            context=context,
            num_samples=num_samples,
            only_return_complete=False,
            max_sample_length=800,
        )
    return vertex_samples


def sample_from_face_model(face_model: pl.LightningModule, context: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Runs face model sampling procedure

    Args:
        face_model: Lightning module with trained weights
        context: Dictionary that contains vertices and masks

    Returns:
        samples: Sampled faces along with masks and other indicator tensors
    """
    face_model.eval()
    with torch.no_grad():
        face_samples = face_model.sample(context=context, max_sample_length=2800)

    return face_samples


def load_vertex_model(config: VertexModelConfig) -> pl.LightningModule:
    """Loads vertex model from config object and .ckpt file

    Args:
        config: VertexModelConfig object

    Returns:
        vertex_model: Initialized vertex model with weights and hyperparameters
    """
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


def load_face_model(config: FaceModelConfig) -> pl.LightningModule:
    """Loads face model from config object and .ckpt file

    Args:
        config: FaceModelConfig object

    Returns:
        face_model: Initialized face model with weights and hyperparameters
    """
    model = config.face_model
    model.load_from_checkpoint(
        FACE_MODEL_CHECKPOINT_FILE,
        encoder_config=model.encoder_config,
        decoder_config=model.decoder_config,
        class_conditional=model.class_conditional,
        num_classes=model.num_classes,
        decoder_cross_attention=model.decoder_cross_attention,
        use_discrete_vertex_embeddings=model.use_discrete_vertex_embeddings,
        quantization_bits=model.quantization_bits,
        max_seq_length=model.max_seq_length,
    )
    return model


def load_config(config_name: str, vertex_config: bool) -> Union[VertexModelConfig, FaceModelConfig]:
    """Loads config object using hydra

    Args:
        config_name: Relative file path to config object
        vertex_config: Whether it's a vertex model config or face model config

    Returns:
        config: A vertex model config or a face model config
    """
    with hydra.initialize_config_module(config_module="polygen.config"):
        cfg = hydra.compose(config_name=config_name)
        if vertex_config:
            config = instantiate(cfg.VertexModelConfig)
        else:
            config = instantiate(cfg.FaceModelConfig)

    return config


def write_vertices_to_obj_files(samples: Dict[str, torch.Tensor], directory_name: str) -> None:
    """Writes generated vertices to .obj files

    Args:
        samples: Generated samples from vertex model
        directory_name: Where to save generated samples
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    vertices = samples["vertices"]
    num_vertices = samples["num_vertices"]
    for i in range(vertices.shape[0]):
        curr_verts = vertices[i, : num_vertices[i]]
        curr_faces = torch.zeros([0, 3]).to(torch.long)
        save_path = os.path.join(directory_name, f"{i}.obj")
        save_obj(save_path, curr_verts, curr_faces)

def write_vertices_and_faces_to_obj(vertex_samples: Dict[str, torch.Tensor], face_samples: Dict[str, torch.Tensor], directory_name: str) -> None:
    """Write generated vertices and faces to .obj files

    Args:
        vertex_samples: Generated samples from vertex model
        face_samples: Generated samples from face model
        directory_name: Where to save generated samples
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    
    vertices = vertex_samples["vertices"]
    num_vertices = vertex_samples["num_vertices"]
    faces = face_samples["faces"]
    num_face_indices = face_samples["num_face_indices"]
    num_objects = vertices.shape[0]
    mesh_list = []
    for i in range(num_objects):
        curr_verts = vertices[i, :num_vertices[i]].numpy()
        curr_faces = faces[i, :num_face_indices[i]].numpy()
        curr_faces = data_utils.unflatten_faces(curr_faces)
        save_path = os.path.join(directory_name, f"{i}.obj")
        data_utils.write_obj(curr_verts, curr_faces, save_path)

def plot_vertices(samples: Dict[str, torch.Tensor]) -> None:
    """Plot generated vertices using matplotlib

    Args:
        samples: Generated samples from vertex model
    """
    vertices = samples["vertices"]
    num_vertices = samples["num_vertices"]
    mesh_list = []
    for i in range(vertices.shape[0]):
        curr_verts = vertices[i, : num_vertices[i]].numpy()
        mesh_list.append({"vertices": curr_verts})

    data_utils.plot_meshes(mesh_list, ax_lims=0.5)
    plt.savefig("generated_meshes.png")


def plot_vertices_and_faces(vertex_samples: Dict[str, torch.Tensor], face_samples: Dict[str, torch.Tensor]) -> None:
    """Plot generated vertices and faces using matplotlib

    Args:
        vertex_samples: samples generated by vertex model
        face_samples: samples generated by face model
    """
    vertices = vertex_samples["vertices"]
    num_vertices = vertex_samples["num_vertices"]
    faces = face_samples["faces"]
    num_face_indices = face_samples["num_face_indices"]
    num_objects = vertices.shape[0]
    mesh_list = []
    for i in range(num_objects):
        curr_verts = vertices[i, : num_vertices[i]].numpy()
        curr_faces = faces[i, : num_face_indices[i]].numpy()
        curr_faces = data_utils.unflatten_faces(curr_faces)
        mesh_list.append({'vertices': curr_verts, 'faces': curr_faces})

    data_utils.plot_meshes(mesh_list, ax_lims = 0.4)
    plt.savefig("generated_meshes_vertices_faces.png")

def test_vertex_model(config_name: str) -> None:
    """Intitializes vertex model and samples from it

    Args:
        config_name: Relative path to config file
    """
    vertex_model_config = load_config(config_name=config_name, vertex_config=True)
    model = load_vertex_model(vertex_model_config)
    context = {"class_label": torch.Tensor([0, 1, 2, 3])}
    samples = sample_from_vertex_model(model, context)
    plot_vertices(samples)


def joint_test_vertex_face_model(vertex_config_name: str, face_config_name: str) -> None:
    """Initializes vertex and face model and joint samples the vertices and faces

    Args:
        vertex_config_name: Vertex model config relative path
        face_config_name: Face model config relative path
    """
    vertex_model_config = load_config(config_name=vertex_config_name, vertex_config=True)
    face_model_config = load_config(config_name=face_config_name, vertex_config=False)
    vertex_model = load_vertex_model(vertex_model_config)
    face_model = load_face_model(face_model_config)
    context = {"class_label": torch.arange(4)}
    vertex_samples = sample_from_vertex_model(vertex_model, context)
    face_samples = sample_from_face_model(face_model, vertex_samples)
    plot_vertices_and_faces(vertex_samples, face_samples)
    write_vertices_and_faces_to_obj(vertex_samples, face_samples, "generated_meshes/")

if __name__ == "__main__":
    pdb.set_trace()
    joint_test_vertex_face_model(vertex_config_name = "vertex_model_config_1231.yaml", face_config_name="face_model_config_1231.yaml")
