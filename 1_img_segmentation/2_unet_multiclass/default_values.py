import torch
import os 

LEARNING_RATE = 1e-4
DEVICE = ("cpu", "cuda")[torch.cuda.is_available()]
BATCH_SIZE = 16
NUM_EPOCHS = 2
SAVE_FREQ = 2
NUM_WORKER = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
N_CLASSES = 5
N_CHANNELS = 3
OUT_CHANNELS = N_CLASSES
PIN_MEMORY = True
LOAD_MODEL = None
VAL_PERCENTAGE = 0.2
AUGMENTATION_P = 0.5
FEATURES = [64, 128, 256, 512, 1024]

DATA_DIR = os.path.abspath(os.path.join(__file__, "../../data/lyft_udacity"))
PATH_TO_SAVE = os.path.abspath(os.path.join(__file__, "../"))