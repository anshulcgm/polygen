import pdb

import torch
import torch.nn as nn

from polygen.modules.data_modules import PolygenDataModule, CollateMethod
from polygen.modules.vertex_model import ImageToVertexModel
from polygen.modules.face_model import FaceModel
import polygen.utils.data_utils as data_utils

img_data_module = PolygenDataModule(data_dir = "image_meshes/", 
                                    collate_method = CollateMethod.IMAGES,
                                    batch_size = 4,
                                    training_split = 1.0,
                                    val_split = 0.0,
                                    use_image_dataset = True,
                                    img_extension = "png",
                                    apply_random_shift_vertices = False,
)

pdb.set_trace()
img_dataset = img_data_module.shapenet_dataset

face_data_module = PolygenDataModule(data_dir = "image_meshes/", 
                                    collate_method = CollateMethod.FACES,
                                    batch_size = 4,
                                    training_split = 1.0,
                                    val_split = 0.0,
                                    use_image_dataset = True,
                                    img_extension = "png",
                                    apply_random_shift_faces = False,
                                    shuffle_vertices = False,
)

img_data_module.setup()
face_data_module.setup()

img_dataloader = img_data_module.train_dataloader()
face_dataloader = face_data_module.train_dataloader()

dataset = img_data_module.shapenet_dataset
mesh_list = []
for i in range(len(dataset)):
    mesh_dict = dataset[i]
    curr_verts, curr_faces = mesh_dict['vertices'], mesh_dict['faces']
    curr_verts = data_utils.dequantize_verts(curr_verts).numpy()
    curr_faces = data_utils.unflatten_faces(curr_faces.numpy())
    mesh_list.append({'vertices': curr_verts, 'faces': curr_faces})

data_utils.plot_meshes(mesh_list, ax_lims = 0.4)

def load_models():
    img_decoder_config = {
        "hidden_size": 256,
        "fc_size": 1024,
        "num_layers": 5,
        'dropout_rate': 0.
    }

    face_transformer_config = {
        'hidden_size': 256,
        'fc_size': 1024,
        'num_layers': 3,
        'dropout_rate': 0.
    }

    img_model = ImageToVertexModel(
        decoder_config = img_decoder_config,
        max_num_input_verts = 800,
        quantization_bits = 8,
    )

    face_model = FaceModel(encoder_config = encoder_config,
                           decoder_config = decoder_config,
                           class_conditional = False,
                           max_seq_length = 500,
                           quantization_bits = 8,
                           decoder_cross_attention = True,
                           use_discrete_vertex_embeddings = True
                          )
    
    return img_model, face_model