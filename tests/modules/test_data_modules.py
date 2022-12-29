"""Tests to ensure that the data modules load .obj files correctly"""
import pdb
import random

import torch

from polygen.modules.data_modules import ShapenetDataset, PolygenDataModule

random.seed(42)
def test_shapenet_dataset():
    dataset = ShapenetDataset(training_dir = "/coc/scratch/aahluwalia30/shapenet/shapenetcore/ShapeNetCore")
    dataset_len = len(dataset)
    rand_index = random.randint(0, dataset_len)
    random_object = dataset[rand_index]