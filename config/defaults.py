from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.NUM_GPU = 1
_C.MODEL.NAME = 'GTZAN'
_C.MODEL.PRE_TRAINED = False
_C.MODEL.PRE_TRAINED_ADDRESS = '../outputs/best_models/final_artist20_slice_3s_model_state_dic.pt'
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
_C.DATALOADER.NUM_WORKERS = 2
_C.DATALOADER.DATASET_ADDRESS = '/home/omid/OMID/projects/python/mldl/NeuralMusicClassification/data/dataset/genres_original'

_C.DATALOADER.NPY_LABELS_TRAIN_DATASET_ADDRESS = '../data/dataset/np_data/train/labels_train_entire_songs.npy'
_C.DATALOADER.NPY_SAMPLES_TRAIN_DATASET_ADDRESS = '../data/dataset/np_data/train/samples_train_entire_songs.npy'

_C.DATALOADER.NPY_LABELS_TEST_DATASET_ADDRESS = '../data/dataset/np_data/test/labels_test.npy'
_C.DATALOADER.NPY_SAMPLES_TEST_DATASET_ADDRESS = '../data/dataset/np_data/test/samples_test.npy'

_C.DATALOADER.NPY_LABELS_VALIDATION_DATASET_ADDRESS = '../data/dataset/np_data/validation/labels_val.npy'
_C.DATALOADER.NPY_SAMPLES_VALIDATION_DATASET_ADDRESS = '../data/dataset/np_data/validation/samples_val.npy'

_C.DATALOADER.LOAD_FROM_NUMPY = True
_C.DATALOADER.ONE_HOT_ENCODING = True
_C.DATALOADER.SR = 16000
_C.DATALOADER.N_MELS = 128,
_C.DATALOADER.N_FFT = 2048
_C.DATALOADER.HOP_LENGTH = 512,

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
_C.OPT.ADAM.WEIGHT_DECAY = 1e-5  # DEFAULT 0
_C.OPT.ADAM.AMS_GRAD = False  # DEFAULT FALSE

# ------------------------------------------------------------------------------ #
_C.OPT.ADADELTA = CN()
_C.OPT.ADADELTA.OPTIMIZER_NAME = "ADADELTA"
_C.OPT.ADADELTA.LR = 0.0001  # DEFAULT  0.001
_C.OPT.ADADELTA.EPS = 1e-08  # DEFAULT 1e-08
_C.OPT.ADADELTA.WEIGHT_DECAY = 1e-5  # DEFAULT 0

# ------------------------------------------------------------------------------ #


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.MAX_EPOCHS = 150

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 16
_C.TEST.WEIGHT = "/Model_GTZAN_70.pth"

# ---------------------------------------------------------------------------- #
# Outputs
# ---------------------------------------------------------------------------- #
_C.DIR = CN()
_C.DIR.OUTPUT_DIR = "../outputs/check_pointers"
_C.DIR.TENSORBOARD_LOG = '../outputs/tensorboard_log'
_C.DIR.BEST_MODEL = '../outputs/best_models'
_C.DIR.FINAL_MODEL = '../outputs/final_model'
