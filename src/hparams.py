import torch
from datetime import datetime


EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 10
LEARNING_RATE = 0.00045040
# Back to standard LR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORKERS = 4 if DEVICE == "cpu" else 4

# -------------------------- Dataset's paths ------------
TEST_DATA_PATH = "../data/fashion-mnist_test.csv"
TRAIN_DATA_PATH = "../data/fashion-mnist_train.csv"
# -------------------------- Tensorboard's path --------
RESULTS_PATH = f"../results/train_{datetime.now().strftime("%m_%d-%H_%M")}/"
FIGURES_PATH = RESULTS_PATH + "assets/"
TENSORBOARD_LOG_DIR = RESULTS_PATH + "fashion_mnist/"

FIGURES_PATH
TENSORBOARD_LOG_DIR
# writer = SummaryWriter(TENSORBOARD_LOG_DIR)
MODEL_PATH = "../models/lenet5_fashion_mnist_0.9127.pth"

STEP = 0
