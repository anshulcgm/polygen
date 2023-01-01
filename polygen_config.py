from typing import Dict

import torch

from polygen.modules.vertex_model import VertexModel
from polygen.modules.face_model import FaceModel
from polygen.modules.data_modules import PolygenDataModule


class VertexModelConfig:
    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        encoder_config: Dict[str, Any],
        class_conditional: bool,
        max_num_input_verts: int,
        quantization_bits: int,
        learning_rate: float,
        step_size: int,
        training_steps: int,
        gamma: float,
        accelerator: str,
    ):
        self.accelerator = accelerator
