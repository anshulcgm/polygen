from typing import Dict, Optional, Tuple, List, Any
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from polygen.utils.data_utils import dequantize_verts

from .polygen_decoder import TransformerDecoder
from .utils import top_k_logits, top_p_logits
from .image_encoder import PolygenResnet


class VertexModel(pl.LightningModule):
    """Autoregressive Generative Model of Quantized Mesh Vertices.
    Operates on flattened vertex sequences with a stopping token:
    [z_0, y_0, x_0, z_1, y_1, x_1, ..., z_n, y_n, x_n, STOP]
    Input Vertex Coordinates are embedded and tagged with learned coordinate and position indicators.
    A transformer decoder outputs logits for a quantized vertex distribution.
    """

    def __init__(
        self,
        decoder_config: Dict[str, Any],
        quantization_bits: int,
        class_conditional: bool = False,
        num_classes: int = 55,
        max_num_input_verts: int = 2500,
        use_discrete_embeddings: bool = True,
        learning_rate: float = 3e-4,
        step_size: int = 5000,
        gamma: float = 0.9995,
    ) -> None:
        """Initializes VertexModel. The encoder can be a model with a Resnet backbone for image contexts and voxel contexts.
        However for class label context, the encoder is simply the class embedder.

        Args:
            decoder_config: Dictionary with TransformerDecoder config. Decoder config has to include num_layers, hidden_size, and fc_size.
            quantization_bits: Number of quantization bits used in mesh preprocessing
            class_conditional: If True, then condition on learned class embeddings
            num_classes: Number of classes to condition on
            max_num_input_verts:  Maximum number of vertices. Used for learned position embeddings.
            use_discrete_embeddings: Discrete embedding layers or linear layers for vertices
            learning_rate: Learning rate for adam optimizer
            step_size: How often to use lr scheduler
            gamma: Decay rate for lr scheduler
        """

        super(VertexModel, self).__init__()
        self.decoder_config = decoder_config
        self.quantization_bits = quantization_bits
        self.embedding_dim = decoder_config["hidden_size"]
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_num_input_verts = max_num_input_verts
        self.use_discrete_embeddings = use_discrete_embeddings
        self.decoder = TransformerDecoder(**decoder_config)
        self.class_embedder = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.embedding_dim)
        self.coord_embedder = nn.Embedding(num_embeddings=3, embedding_dim=self.embedding_dim)
        self.pos_embedder = nn.Embedding(num_embeddings=self.max_num_input_verts, embedding_dim=self.embedding_dim)
        self.vert_embedder_discrete = nn.Embedding(
            num_embeddings=2 ** self.quantization_bits + 1,
            embedding_dim=self.embedding_dim,
        )
        self.linear_layer = nn.Linear(self.embedding_dim, 2 ** self.quantization_bits + 1)

        zero_embeddings_tensor = torch.randn([1, 1, self.embedding_dim], device=self.device)
        self.zero_embed = nn.Parameter(zero_embeddings_tensor)

        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma

    def _embed_class_label(self, labels: torch.Tensor) -> torch.Tensor:
        """Embeds Class Label with learned embedding matrix

        Args:
            labels: A Tensor with shape [batch_size,]. Represents the class label for each sample in the batch.
        Returns:
            embeddings: A Tensor with shape [batch_size, embed_size].
        """
        return self.class_embedder(labels.to(torch.int64))

    def _prepare_context(self, context: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepares global context embedding

        Args:
            context: A dictionary that contains a key of class_label
        Returns:
            global_context_embeddings: A Tensor of shape [batch_size, embed_size]
            sequential_context_embeddings: None
        """
        if self.class_conditional:
            global_context_embedding = self._embed_class_label(context["class_label"])
        else:
            global_context_embedding = None
        return global_context_embedding, None

    def _embed_inputs(self, vertices: torch.Tensor, global_context_embedding: torch.Tensor = None) -> torch.Tensor:
        """
        Embeds flat vertices and adds position and coordinate information.

        Args:
            vertices: A Tensor of shape [batch_size, sample_length]. Represents current sampled vertices.
            global_context_embedding: A Tensor of shape [batch_size, embed_size]. Represents class label conditioning.
        Returns:
            embeddings: A Tensor of shape [sample_length + 1, batch_size]. Represents combination of embeddings with global context embeddings. The first and second
                        dimensions are transposed for the sake of the decoder.
        """
        input_shape = vertices.shape
        batch_size, seq_length = input_shape[0], input_shape[1]
        coord_embeddings = self.coord_embedder(
            torch.fmod(torch.arange(seq_length, device=self.device), 3)
        )  # Coord embeddings will be of shape [seq_length, embed_size]
        pos_embeddings = self.pos_embedder(
            torch.floor_divide(torch.arange(seq_length, device=self.device), 3)
        )  # Position embeddings will be of shape [seq_length, embed_size]
        vert_embeddings = self.vert_embedder_discrete(
            vertices
        )  # Vert embeddings will be of shape [batch_size, seq_length, embed_size]
        if global_context_embedding is None:
            zero_embed_tiled = torch.repeat_interleave(self.zero_embed, batch_size, dim=0)
        else:
            zero_embed_tiled = global_context_embedding[:, None].to(
                torch.float32
            )  # Zero embed tiled is of shape [batch_size, 1, embed_size]

        embeddings = vert_embeddings + (coord_embeddings + pos_embeddings)[None]

        # Embeddings shape before concatenation is [batch_size, seq_length, embed_size], after concatenation it is [batch_size, seq_length + 1, embed_size]
        embeddings = torch.cat([zero_embed_tiled, embeddings], dim=1)

        # Changing the dimension from [batch_size, seq_length, embed_size] to [seq_length, batch_size, embed_size] for TransformerDecoder
        return embeddings.transpose(0, 1)

    def _project_to_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs decoder outputs through a linear layer

        Args:
            inputs: Tensor of shape [batch_size, sequence_length, embed_size].
        Returns:
            outputs: Tensor of shape [batch_size, sequence_length, 2 ** self.quantization_bits + 1].
        """
        output = self.linear_layer(inputs)
        return output

    def _create_dist(
        self,
        vertices: torch.Tensor,
        global_context_embedding: Optional[torch.Tensor] = None,
        sequential_context_embedding: Optional[torch.Tensor] = None,
        cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: int = 1,
    ) -> torch.Tensor:
        """Creates a predictive distribution for the next vertex sample

        Args:
            vertices: A Tensor of shape [batch_size, sequence_length]. Represents current flattened vertices. Sequence length is at max 3 * the number of vertices.
            global_context_embedding: A Tensor of shape [batch_size, embed_size]. Represents conditioning on class labels.
            sequential_context_embeddings: A Tensor of shape [batch_size, context_seq_length, context_embed_size]. Represents conditioning on images or voxels.
            cache:  A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Each dictionary in the list represents the cache at the respective decoder layer.
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to take out for top-k sampling.
            top-p: Proportion of probability mass to keep for top-p sampling.
        Returns:
            logits: Logits that can be used to create a categorical distribution to sample the next vertex.
        """
        decoder_inputs = self._embed_inputs(vertices.to(torch.int64), global_context_embedding)
        if cache is not None:
            decoder_inputs = decoder_inputs[-1:, :]
        if sequential_context_embedding is not None:
            sequential_context_embedding = sequential_context_embedding.transpose(0, 1)
        outputs = self.decoder(
            decoder_inputs,
            sequential_context_embeddings=sequential_context_embedding,
            cache=cache,
        ).transpose(
            0, 1
        )  # Transpose to convert from [seq_length, batch_size, embedding_dim] to [batch_size, seq_length, embedding_dim]
        # pass through linear layer
        logits = self._project_to_logits(outputs)
        logits = logits / temperature
        # remove the smaller logits
        logits = top_k_logits(logits, top_k)
        # then choose those that contribute to 90% of mass distribution
        logits = top_p_logits(logits, top_p)
        return logits

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward method for Vertex Model
        Args:
            batch: A dictionary with a key of vertices_flat that represents a flattened input sequence of vertices.
        Returns:
            logits: Logits that can be used to create a categorical distribution to sample the next vertex.
        """
        global_context, seq_context = self._prepare_context(batch)
        vertices = batch["vertices_flat"]
        logits = self._create_dist(
            vertices[:, :-1],
            global_context_embedding=global_context,
            sequential_context_embedding=seq_context,
        )
        return logits

    def training_step(self, vertex_model_batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.float32:
        """Pytorch Lightning training step method

        Args:
            vertex_model_batch: A dictionary that contains the flat vertices
            batch_idx: Which batch are we processing

        Returns:
            vertex_loss: NLL loss for estimated categorical distribution
        """
        vertex_logits = self(vertex_model_batch)
        vertex_pred_dist = torch.distributions.categorical.Categorical(logits=vertex_logits)
        vertex_loss = -torch.sum(
            vertex_pred_dist.log_prob(vertex_model_batch["vertices_flat"]) * vertex_model_batch["vertices_flat_mask"]
        )
        self.log("train_loss", vertex_loss)
        return vertex_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Method to create optimizer and learning rate scheduler

        Returns:
            dict: A dictionary with optimizer and learning rate scheduler
        """
        vertex_model_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        vertex_model_scheduler = torch.optim.lr_scheduler.StepLR(
            vertex_model_optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return {
            "optimizer": vertex_model_optimizer,
            "lr_scheduler": vertex_model_scheduler,
        }

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.float32:
        """Validation step for Pytorch Lightning

        Args:
            val_batch: dictionary which contains batch to run validation on
            batch_idx: Which batch we are processing

        Returns:
            vertex_loss: NLL loss for estimated categorical distribution
        """
        with torch.no_grad():
            vertex_logits = self(val_batch)
            vertex_pred_dist = torch.distributions.categorical.Categorical(logits=vertex_logits)
            vertex_loss = -torch.sum(
                vertex_pred_dist.log_prob(val_batch["vertices_flat"]) * val_batch["vertices_flat_mask"]
            )
        self.log("val_loss", vertex_loss)
        return vertex_loss

    def sample(
        self,
        num_samples: int,
        max_sample_length: int = 50,
        context: Dict[str, torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        recenter_verts: bool = True,
        only_return_complete: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive sampling method to generate vertices

        Args:
            num_samples: Number of samples to produce.
            context: A dictionary with the type of context to condition upon. This could be class labels or images or voxels.
            max_sample_length: Maximum length of sampled vertex samples. Sequences that do not complete are truncated.
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to keep for top-k sampling.
            top-p: Proportion of probability mass to keep for top-p sampling.
            recenter_verts: If True, center vertex samples around origin. This should be used if model is trained using shift augmentations.
            only_return_complete: If True, only return completed samples. Otherwise return all samples along with completed indicator.

        Returns:
            outputs: Output dictionary with fields
                'completed': Boolean tensor of shape [num_samples]. If True then corresponding sample completed within max_sample_length.
                'vertices': Tensor of samples with shape [num_samples, num_verts, 3].
                'num_vertices': Tensor indicating number of vertices for each example in padded vertex samples.
                'vertices_mask': Tensor of shape [num_samples, num_verts] that masks corresponding invalid elements in vertices.
        """
        global_context, seq_context = self._prepare_context(context)

        # limit context shape to number of samples desired
        if global_context is not None:
            num_samples = min(num_samples, global_context.shape[0])
            global_context = global_context[:num_samples]
            if seq_context is not None:
                seq_context = seq_context[:num_samples]
        elif seq_context is not None:
            num_samples = min(num_samples, seq_context.shape[0])
            seq_context = seq_context[:num_samples]

        def _loop_body(
            i: int,
            samples: torch.Tensor,
            cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        ) -> Tuple[int, torch.Tensor]:
            """While-loop body for autoregression calculation.

            Args:
                i: Current iteration in the loop
                samples: tensor of shape [num_samples, i].
                cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}.
                       Each dictionary in the list represents the cache at the respective decoder layer.
            Returns:
                next_iter: i + 1.
                samples: tensor of shape [num_samples, i + 1] or of shape [num_samples, 2 * i + 1] if cache doesn't exist.
            """
            logits = self._create_dist(
                samples,
                global_context_embedding=global_context,
                sequential_context_embedding=seq_context,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            next_sample = cat_dist.sample()
            samples = torch.cat([samples, next_sample.to(torch.int32)], dim=1)
            return i + 1, samples

        def _stopping_cond(samples: torch.Tensor) -> bool:
            """
            Stopping condition for sampling while-loop. Looking for stop token (represented by 0)
            Args:
                samples: tensor of shape [num_samples, i], where i is the current iteration.
            Returns:
                token_found: Boolean that represents if stop token has been found.
            """
            nonzero_boolean_matrix = torch.ne(samples, 0)
            reduced_matrix = torch.all(
                nonzero_boolean_matrix, dim=-1
            )  # Checks if stopping token is present in every row
            return torch.any(reduced_matrix)

        samples = torch.zeros([num_samples, 0], dtype=torch.int32)
        cache = self.decoder.initialize_cache(num_samples)
        max_sample_length = max_sample_length or self.max_num_input_verts
        j = 0
        while _stopping_cond(samples) and j < max_sample_length * 3 + 1:
            j, samples = _loop_body(j, samples, cache)

        completed_samples_boolean = samples == 0  # Checks for stopping token
        completed = torch.any(
            completed_samples_boolean, dim=-1
        )  # Indicates which samples are completed of shape [num_samples,]
        stop_index_completed = torch.argmax(completed_samples_boolean.to(torch.int32), dim=-1).to(
            torch.int32
        )  # Indicates where the stopping token occurs in each batch of samples
        stop_index_incomplete = (
            max_sample_length * 3 * torch.ones_like(stop_index_completed)
        )  # Placeholder tensor used to select samples from incomplete samples
        stop_index = torch.where(
            completed, stop_index_completed, stop_index_incomplete
        )  # Stopping Indices of each sample, if completed is true, then stopping index is taken from completed stop index tensor
        num_vertices = torch.floor_divide(stop_index, 3)

        samples = samples[:, : (torch.max(num_vertices) * 3)] - 1  # Selects last possible stopping index
        verts_dequantized = dequantize_verts(samples, self.quantization_bits)
        # Converts vertices to [-1, 1] range
        vertices = torch.reshape(verts_dequantized, [num_samples, -1, 3])  # Reshapes into 3D Tensors
        vertices = torch.stack(
            [vertices[..., 2], vertices[..., 1], vertices[..., 0]], dim=-1
        )  # Converts from z-y-x to x-y-z.

        # Pad samples such that samples of different lengths can be concatenated
        pad_size = max_sample_length - vertices.shape[1]
        vertices = F.pad(vertices, [0, 0, 0, pad_size, 0, 0])

        vertices_mask = (torch.arange(max_sample_length)[None] < num_vertices[:, None]).to(
            torch.float32
        )  # Provides a mask of which vertices to zero out as they were produced after stop token for that batch ended

        if recenter_verts:
            vert_max, _ = torch.max(vertices - 1e10 * (1.0 - vertices_mask)[..., None], dim=1, keepdim=True)
            vert_min, _ = torch.min(vertices + 1e10 * (1.0 - vertices_mask)[..., None], dim=1, keepdim=True)
            vert_centers = 0.5 * (vert_max + vert_min)
            vertices = vertices - vert_centers

        vertices = vertices * vertices_mask[..., None]  # Zeros out vertices produced after stop token

        if only_return_complete:
            vertices = vertices[completed]
            num_vertices = num_vertices[completed]
            vertices_mask = vertices_mask[completed]
            completed = completed[completed]

        outputs = {
            "completed": completed,
            "vertices": vertices,
            "num_vertices": num_vertices,
            "vertices_mask": vertices_mask.to(torch.int32),
        }
        return outputs


class ImageToVertexModel(VertexModel):
    def __init__(
        self,
        decoder_config: Dict[str, Any],
        quantization_bits: int,
        use_discrete_embeddings: bool = True,
        max_num_input_verts: int = 2500,
    ) -> None:
        """Initializes the resnet module along with an embedder

        Args:
            decoder_config: Dictionary with TransformerDecoder config. Decoder config has to include num_layers, hidden_size, and fc_size.
            quantization_bits: Number of quantization bits used in mesh preprocessing
            use_discrete_embeddings: Whether or not we're working with quantized vertices
            max_num_input_verts: Maximum number of vertices. Used for learned position embeddings.
        """
        super(ImageToVertexModel, self).__init__(
            decoder_config=decoder_config,
            quantization_bits=quantization_bits,
            max_num_input_verts=max_num_input_verts,
            use_discrete_embeddings=use_discrete_embeddings,
        )
        self.res_net = PolygenResnet()
        self.embedder = nn.Linear(2, self.embedding_dim)

    def _prepare_context(self, context: Dict[str, torch.Tensor]) -> Tuple[None, torch.Tensor]:
        """Creates image embeddings using resnet and flattened image

        Args:
            context: A dictionary that contains an image

        Returns:
            sequential_context_embeddings: Processed image embeddings
        """
        image_embeddings = self.res_net(context["image"] - 0.5)
        image_embeddings = image_embeddings.permute(0, 2, 3, 1)
        processed_image_resolution = image_embeddings.shape[1:3]
        x = torch.linspace(-1, 1, processed_image_resolution[0])
        y = torch.linspace(-1, 1, processed_image_resolution[1])
        image_coords = torch.stack(torch.meshgrid(x, y), dim=-1)
        image_coord_embeddings = self.embedder(image_coords)
        image_embeddings = image_embeddings + image_coord_embeddings[None]
        batch_size = image_embeddings.shape[0]
        sequential_context_embeddings = torch.reshape(image_embeddings, [batch_size, -1, self.embedding_dim])

        return None, sequential_context_embeddings
