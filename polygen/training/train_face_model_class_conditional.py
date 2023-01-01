import torch
import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate

from polygen.polygen_config import FaceModelConfig


def main() -> None:
    with hydra.initialize_config_module(config_module="polygen.config"):
        cfg = hydra.compose(config_name="face_model_config_1231.yaml")
        face_model_config = instantiate(cfg.FaceModelConfig)

    face_data_module = face_model_config.face_data_module
    face_model = face_model_config.face_model

    training_steps = face_model_config.training_steps
    batch_size = face_model_config.batch_size
    dataset_length = len(face_data_module.shapenet_dataset)
    num_epochs = training_steps * batch_size // (dataset_length)
    
    trainer = pl.Trainer(
        accelerator=face_model_config.accelerator, 
        gpus=face_model_config.num_gpus, 
        max_epochs=num_epochs
    )
    trainer.fit(face_model, face_data_module)


if __name__ == "__main__":
    main()