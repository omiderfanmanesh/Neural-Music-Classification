from yacs.config import CfgNode as CN
import torch
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.NUM_GPU = 1
_C.MODEL.BASED_MODEL = True
_C.MODEL.LOAD_MODEL = False
_C.MODEL.MODEL_TO_LOAD = "C:\\Users\\Giovanni Calleris\\Desktop\\Phyton\\PyCharm\\NeuralMusicClassification\\music_checkpoint_506_std_2_910.pt" # music_checkpoint_506_std_2_910 with basdemodel as dataset
_C.MODEL.FREQ_RANGE = 64 # for the extension
_C.MODEL.SCALE = True
_C.MODEL.MEAN = 0
_C.MODEL.STD = 1
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.TIME = 1024
_C.INPUT.MEL = 128
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.DATASET_ADDRESS = 'C:\\Users\\Giovanni Calleris\\Desktop\\Phyton\\PyCharm\\NeuralMusicClassification\\data\dataset\\genres_original'
_C.DATALOADER.NPY_SAMPLES_TRAINING_DATASET_ADDRESS = 'C:\\Users\\Giovanni Calleris\\Desktop\\Phyton\\PyCharm\\NeuralMusicClassification\\data\\dataset\\basedmodel_corretto_kamikaze\\genres_spectrogram_training'
_C.DATALOADER.NPY_SAMPLES_TEST_DATASET_ADDRESS = 'C:\\Users\\Giovanni Calleris\\Desktop\\Phyton\\PyCharm\\NeuralMusicClassification\\data\\dataset\\basedmodel_corretto_kamikaze\\genres_spectrogram_test'
_C.DATALOADER.NPY_SAMPLES_VALIDATION_DATASET_ADDRESS = 'C:\\Users\\Giovanni Calleris\\Desktop\\Phyton\\PyCharm\\NeuralMusicClassification\\data\\dataset\\basedmodel_corretto_kamikaze\\genres_spectrogram_validation'
_C.DATALOADER.LOAD_FROM_NUMPY = True
_C.DATALOADER.ONE_HOT_ENCODING = True
_C.DATALOADER.SR = 16000
_C.DATALOADER.N_MELS = 128,
_C.DATALOADER.N_FFT = 2048
_C.DATALOADER.HOP_LENGTH = 512,
_C.DATALOADER.SONG_LENGTH = 94

# ---------------------------------------------------------------------------- #
# Optimizers
# ---------------------------------------------------------------------------- #
_C.OPT = CN()

_C.OPT.SGD = CN()

_C.OPT.SGD.OPTIMIZER_NAME = "SGD"
_C.OPT.SGD.LR = 0.001  # DEFAULT  0.001
_C.OPT.SGD.MOMENTUM = 0  # DEFAULT 0
_C.OPT.SGD.WEIGHT_DECAY = 0  # DEFAULT 0
_C.OPT.SGD.DAMPENING = 0  # DEFAULT 0
_C.OPT.SGD.NESTEROV = False  # DEFAULT FALSE

# ------------------------------------------------------------------------------ #
_C.OPT.ADAM = CN()
_C.OPT.ADAM.OPTIMIZER_NAME = "ADAM"
_C.OPT.ADAM.LR = 0.0001  # DEFAULT  0.001
_C.OPT.ADAM.BETAS = [0.9, 0.999]  # DEFAULT [0.9, 0.999]
_C.OPT.ADAM.EPS = 1e-08  # DEFAULT 1e-08
_C.OPT.ADAM.WEIGHT_DECAY = 0.00008  # DEFAULT 0
_C.OPT.ADAM.AMS_GRAD = False  # DEFAULT FALSE

# ------------------------------------------------------------------------------ #


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.MAX_EPOCHS = 50



_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.IMS_PER_BATCH_VAL_AND_TEST = 10

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 32
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "../outputs"