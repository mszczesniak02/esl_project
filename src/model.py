import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
from hparams import *
from dataloader import *


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,
                               padding=2)  # 28x28 -> 28x28
        # 14x14 -> 10x10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 10x10 -> 5x5

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))  # 84-dim embeddings for visualization
        x = self.fc3(features)

        if return_features:
            return x, features
        return x


def model_set(lrate=LEARNING_RATE) -> tuple[LeNet5, nn.CrossEntropyLoss, optim.Adam]:
    model = LeNet5(num_classes=NUM_CLASSES).to(DEVICE)
    print(f" Model LeNet5 on: {DEVICE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lrate)
    criterion = nn.CrossEntropyLoss()

    print(f"    Loss function: CrossEntropyLoss")
    print(f"     Optimizer LR:{lrate}")

    return model, criterion, optimizer


def model_load(path=MODEL_PATH, device=DEVICE) -> LeNet5:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = LeNet5()

    model.load_state_dict(torch.load(
        path, map_location=device, weights_only=False))
    model = model.to(device)

    return model


def model_load_quantized(path, device='cpu'):
    """
    Load a quantized model (int8).

    Note: Quantized models must run on CPU, not CUDA.

    Args:
        path: Path to quantized model (.pth file)
        device: Must be 'cpu' for quantized models

    Returns:
        Quantized LeNet5 model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if device != 'cpu':
        print(f"⚠️  Warning: Quantized models only work on CPU, changing device to 'cpu'")
        device = 'cpu'

    # Create base model
    model_fp32 = LeNet5()
    model_fp32.eval()

    # Quantize it (this creates the right architecture)
    import torch.quantization
    model_quantized = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )

    # Now load the saved quantized weights
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")
        model_quantized.load_state_dict(torch.load(
            path, map_location=device, weights_only=False))

    model_quantized = model_quantized.to(device)
    model_quantized.eval()

    return model_quantized
