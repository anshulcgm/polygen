from typing import Any, Dict

import torch

from polygen.modules.vertex_model import VertexModel
from polygen.modules.face_model import FaceModel
from polygen.modules.data_modules import PolygenDataModule, CollateMethod


class VertexModelConfig:
    def __init__(
        self,
        accelerator: str,
        dataset_path: str,
        batch_size: int,
        training_split: float,
        val_split: float,
        apply_random_shift: bool,
        decoder_config: Dict[str, Any],
        quantization_bits: int,
        class_conditional: bool,
        num_classes: int,
        max_num_input_verts: int,
        use_discrete_embeddings: bool,
        learning_rate: float,
        step_size: int,
        gamma: float,
        training_steps: int,
    ) -> None:
        """Initializes vertex model and vertex data module

        Args:
            accelerator: data parallel or distributed data parallel
            dataset_path: Root directory for shapenet dataset
            batch_size: How many 3D objects in one batch
            training_split: What proportion of data to use for training the model
            val_split: What proportion of data to use for validation
            apply_random_shift: Whether or not we're applying random shift to vertices
            decoder_config: Dictionary with TransformerDecoder config. Decoder config has to include num_layers, hidden_size, and fc_size.
            quantization_bits: Number of quantization bits used in mesh preprocessing
            class_conditional: If True, then condition on learned class embeddings
            num_classes: Number of classes to condition on
            max_num_input_verts:  Maximum number of vertices. Used for learned position embeddings.
            use_discrete_embeddings: Discrete embedding layers or linear layers for vertices
            learning_rate: Learning rate for adam optimizer
            step_size: How often to use lr scheduler
            gamma: Decay rate for lr scheduler
            training_steps: How many total steps we want to train for
        """

        num_gpus = torch.cuda.device_count()
        self.accelerator = accelerator
        if accelerator.startswith("ddp"):
            self.batch_size = batch_size // num_gpus
        else:
            self.batch_size = batch_size

        self.data_module = PolygenDataModule(
            data_dir=dataset_path,
            batch_size=self.batch_size,
            collate_method=CollateMethod.VERTICES,
            training_split=training_split,
            val_split=val_split,
            quantization_bits=quantization_bits,
            apply_random_shift_vertices=apply_random_shift,
        )

        self.vertex_model = VertexModel(
            decoder_config=decoder_config,
            quantization_bits=quantization_bits,
            class_conditional=class_conditional,
            num_classes=num_classes,
            max_num_input_verts=max_num_input_verts,
            use_discrete_embeddings=use_discrete_embeddings,
            learning_rate=learning_rate,
            step_size=step_size,
            gamma=gamma,
        )

        self.training_steps = training_steps
