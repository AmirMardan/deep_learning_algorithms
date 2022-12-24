import torch
from torch import Tensor
from typing import Any, List, Optional, Union
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

LEARNING_RATE = 1e-4
DEVICE = ("cpu", "cuda")[torch.cuda.is_available()]
BATCH_SIZE = 16
NUM_EPOCHS = 2
SAVE_FREQ = 2
NUM_WORKER = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
N_CLASSES = 1
N_CHANNELS = 3
PIN_MEMORY = True
LOAD_MODEL = None
VAL_PERCENTAGE = 0.1
AUGMENTATION_P = 0.5
FEATURES = [64, 128, 256, 512, 1024]

