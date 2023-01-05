import pdb

import torch
import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate

from polygen.polygen_config import VertexModelConfig


def main(config_name: str) -> None:
    with hydra.initialize_config_module(config_module="polygen.config"):
        cfg = hydra.compose(config_name=config_name)
        vertex_model_config = instantiate(cfg.VertexModelConfig)

    vertex_data_module = vertex_model_config.vertex_data_module
    vertex_model = vertex_model_config.vertex_model

    training_steps = vertex_model_config.training_steps
    batch_size = vertex_model_config.batch_size
    dataset_length = len(vertex_data_module.shapenet_dataset)
    num_epochs = training_steps * batch_size // (dataset_length)

    trainer = pl.Trainer(
        accelerator=vertex_model_config.accelerator,
        gpus=vertex_model_config.num_gpus,
        max_epochs=num_epochs,
    )
    trainer.fit(vertex_model, vertex_data_module)


if __name__ == "__main__":
    # main(config_name = "vertex_model_config_1231.yaml") #To train class conditioned model
    main(config_name="image_model_config_105.yaml")  # To train image conditioned model
