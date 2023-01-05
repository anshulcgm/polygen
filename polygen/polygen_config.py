from typing import Any, Dict, Optional

import torch

from polygen.modules.vertex_model import VertexModel, ImageToVertexModel
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
        image_model: bool = False,
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
            image_model: Whether we're training the image model or class-conditioned model
        """

        self.num_gpus = torch.cuda.device_count()
        self.accelerator = accelerator
        if accelerator.startswith("ddp"):
            self.batch_size = batch_size // self.num_gpus
        else:
            self.batch_size = batch_size

        if image_model:
            collate_method = CollateMethod.IMAGES
            self.vertex_model = ImageToVertexModel(
                decoder_config = decoder_config, 
                quantization_bits = quantization_bits,
                use_discrete_embeddings = use_discrete_embeddings,
                max_num_input_verts = max_num_input_verts,
                learning_rate = learning_rate,
                step_size = step_size,
                gamma = gamma,
            )
        else:
            collate_method = CollateMethod.VERTICES
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


        self.vertex_data_module = PolygenDataModule(
            data_dir=dataset_path,
            batch_size=self.batch_size,
            collate_method=collate_method,
            training_split=training_split,
            val_split=val_split,
            quantization_bits=quantization_bits,
            use_image_dataset = image_model,
            apply_random_shift_vertices=apply_random_shift,
        )

        self.training_steps = training_steps

class FaceModelConfig:
    def __init__(
        self,
        accelerator: str,
        dataset_path: str,
        batch_size: int,
        training_split: float,
        val_split: float,
        apply_random_shift: bool,
        shuffle_vertices: bool,
        encoder_config: Dict,
        decoder_config: Dict,
        class_conditional: bool,
        num_classes: int,
        decoder_cross_attention: bool,
        use_discrete_vertex_embeddings: bool,
        quantization_bits: int,
        max_seq_length: int,
        learning_rate: float,
        step_size: int,
        gamma: float,
        training_steps: int,
    ):
        """Initializes face model and face data module

        Args:
            accelerator: data parallel or distributed data parallel
            dataset_path: Root directory for shapenet dataset
            batch_size: How many 3D objects in one batch
            training_split: What proportion of data to use for training the model
            val_split: What proportion of data to use for validation
            apply_random_shift: Whether or not we're applying random shift to vertices
            shuffle_vertices: Whether or not we are randomly shuffling the vertices during batch generation
            encoder_config: Dictionary representing config for PolygenEncoder
            decoder_config: Dictionary representing config for TransformerDecoder
            class_conditional: If we are using global context embeddings based on class labels
            num_classes: How many distinct classes in the dataset
            decoder_cross_attention: If we are using cross attention within the decoder
            use_discrete_vertex_embeddings: Are the inputted vertices quantized
            quantization_bits: How many bits are we using to encode the vertices
            max_seq_length: Max number of face indices we can generate
            learning_rate: Learning rate for adam optimizer
            step_size: How often to use lr scheduler
            gamma: Decay rate for lr scheduler
            training_steps: How many total steps we want to train for
        """

        self.num_gpus = torch.cuda.device_count()
        self.accelerator = accelerator
        if accelerator.startswith("ddp"):
            self.batch_size = batch_size // self.num_gpus
        else:
            self.batch_size = batch_size

        self.face_data_module = PolygenDataModule(
            data_dir = dataset_path,
            batch_size = self.batch_size,
            collate_method = CollateMethod.FACES,
            training_split = training_split,
            val_split = val_split,
            quantization_bits = quantization_bits,
            apply_random_shift_faces = apply_random_shift,
            shuffle_vertices = shuffle_vertices,
        )

        self.face_model = FaceModel(
            encoder_config = encoder_config,
            decoder_config = decoder_config,
            class_conditional = class_conditional,
            num_classes = num_classes,
            decoder_cross_attention = decoder_cross_attention,
            use_discrete_vertex_embeddings = use_discrete_vertex_embeddings,
            quantization_bits = quantization_bits,
            max_seq_length = max_seq_length,
            learning_rate = learning_rate,
            step_size = step_size,
            gamma = gamma,
        )
        
        self.training_steps = training_steps

