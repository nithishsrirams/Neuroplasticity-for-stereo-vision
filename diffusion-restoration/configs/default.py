import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

KITTI_ZIP_PATH = os.path.join(PROJECT_ROOT, "data_scene_flow.zip")
KITTI_EXTRACT_PATH = os.path.join(PROJECT_ROOT, "kitti_data")
PSMNET_DIR = os.path.join(BASE_DIR, "PSMNet")

CROP_H = 256
CROP_W = 512
BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-4
MAX_DISP = 192

N_DISP_BINS = 64
CE_SIGMA = 3.0
USE_FEATURE_GATE = True
FEAT_CHANNELS = 32

TRAIN_DEGRADE_PROB = 0.7
DEGRADE_PROB = 1.0
DEGRADE_BASE_SEED = 12345
DEGRADE_CAMERA = "left"
DEGRADE_TYPE = "noise"
DEGRADE_SEVERITY = 3

RAW_SUP_WEIGHT = 0.25
SELF_SUP_WEIGHT = 0.50
GATE_LOSS_WEIGHT = 0.10
DIFF_LOSS_WEIGHT = 0.10

SEED = 42
TRAIN_SPLIT = 0.9

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "diffusion_restoration_history.json")
SPLIT_PATH = os.path.join(BASE_DIR, "diffusion_restoration_split.json")
TEST_PRED_DIR = os.path.join(OUTPUT_DIR, "diffusion_restoration_test_predictions")

PRETRAINED_BACKBONE_PATH = os.path.join(PSMNET_DIR, "pretrained_model_KITTI2015.tar")

RUN_KITTI_VISUALIZATION = False
RUN_DEGRADED_VISUALIZATION = True
EVAL_ON_CLEAN_AND_DEGRADED = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_BEST = "diffusion_restoration_best_kitti.pth"
