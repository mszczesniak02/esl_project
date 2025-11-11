
import torch.nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hparams import *


class FashionMNISTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row.iloc[0])
        pixels = row.iloc[1:].values.astype(np.float32)
        image = pixels.reshape(28, 28)

        # Normalize to [0, 1]
        image = image / 255.0

        if self.transform:
            image = self.transform(image)

        # Add channel dimension: (28, 28) -> (1, 28, 28)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def dataset_visualize(dataset: pd.DataFrame, file_title="dataset_visualize.png", file_path=FIGURES_PATH) -> None:
    """
    Visualize the dataset, show a random picure of each category from a selected dataset

    Parameters
    ----------
    dataset : pd.DataFrame dataset to show a the pictures from

    """

    label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                   "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    fig, axes = plt.subplots(2, 5, figsize=(12, 8))
    axes = axes.flatten()

    label_col = dataset.columns[0]
    labels = sorted(dataset[label_col].unique())[:10]

    for i, label in enumerate(labels):
        subset = dataset[dataset[label_col] == label]
        if subset.empty:
            axes[i].axis('off')
            continue

        sample = subset.sample(n=1).iloc[0]
        pixels = sample.iloc[1:].values
        img = pixels.reshape(28, 28).astype(np.uint8)

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {label} ({label_names[label]})")
        axes[i].axis('off')

    plt.tight_layout(h_pad=0)
    plt.savefig(file_path + file_title)


def dataset_load(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH, bsize=BATCH_SIZE, epochs=EPOCHS, print_params=False) -> tuple[DataLoader, DataLoader]:
    """
    Load the CSV data into memory and convert it into DataLoader with global hyperparameters  
    """
    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    train_ds = FashionMNISTDataset(train_dataset)
    test_ds = FashionMNISTDataset(test_dataset)
    #
    train_loader = DataLoader(
        train_ds, batch_size=bsize, shuffle=True, num_workers=WORKERS)
    test_loader = DataLoader(test_ds, batch_size=bsize,
                             shuffle=False, num_workers=WORKERS)
    #
    if print_params:
        print(f"          Epochs: {epochs}")
        print(f"      Batch size: {bsize}")
        print(f"   Train samples: {len(train_ds)}")
        print(f"    Test samples: {len(test_ds)}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"    Test batches: {len(test_loader)}")

    return train_loader, test_loader


""""
dane są podzielne, 
80% 20%

hyperparametry dostronjone

trening modelu na 45 epokach do 91,27%
infrerence                              - 
pruning                                 - ucinanie struktur zamiast samych wag
kwantyacja                              - zmiana na int8


"""
