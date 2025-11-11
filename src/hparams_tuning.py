import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import make_grid

# ---------------


# import albumentations as A
# ============================================================================


def plot_confusion_matrix(model, loader, device, class_names, writer, global_step, tag='confusion_matrix'):
    """
    Generate and log confusion matrix to TensorBoard
    Shows only raw counts for clarity
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Computing Confusion Matrix', leave=False):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Create figure with only raw counts
    fig, ax = plt.subplots(figsize=(12, 10))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('True', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Confusion Matrix - Epoch {global_step}', fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Log to TensorBoard as figure
    writer.add_figure(tag, fig, global_step)
    plt.close(fig)

    return cm


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


# =======================HYPERPARAMETERS=====================================
EPOCHS = 10
BATCH_SIZE = 16
NUM_CLASSES = 10
LEARNING_RATE = 0.001  # Back to standard LR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORKERS = 4 if DEVICE == "cpu" else 4

# -------------------------- Dataset's paths ------------
TEST_DATA_PATH = "data/fashion-mnist_test.csv"
TRAIN_DATA_PATH = "data/fashion-mnist_train.csv"
# -------------------------- Tensorboard's path --------
RESULTS_PATH = f"results/train_{datetime.now().strftime("%m_%d-%H_%M")}/"
FIGURES_PATH = RESULTS_PATH + "assets/"
TENSORBOARD_LOG_DIR = RESULTS_PATH + "fashion_mnist/"

FIGURES_PATH
TENSORBOARD_LOG_DIR
# writer = SummaryWriter(TENSORBOARD_LOG_DIR)

STEP = 0

# ============================================================================


def dataset_visualize(dataset: pd.DataFrame, file_title="dataset_visualize.png", file_path=FIGURES_PATH) -> None:
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


def dataset_load(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH, bsize=BATCH_SIZE, epochs=EPOCHS, ) -> tuple[DataLoader, DataLoader]:
    """
    Load the CSV data into memory and convert it into DataLoader with global hyperparameters  
    """
    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    # dataset_visualize(train_dataset, "train_dataset.png")
    # dataset_visualize(test_dataset, "test_dataset.png")
    #
    train_ds = FashionMNISTDataset(train_dataset)
    test_ds = FashionMNISTDataset(test_dataset)
    #
    train_loader = DataLoader(
        train_ds, batch_size=bsize, shuffle=True, num_workers=WORKERS)
    test_loader = DataLoader(test_ds, batch_size=bsize,
                             shuffle=False, num_workers=WORKERS)
    #
    # print(f"          Epochs: {epochs}")
    # print(f"      Batch size: {bsize}")
    # print(f"   Train samples: {len(train_ds)}")
    # print(f"    Test samples: {len(test_ds)}")
    # print(f"   Train batches: {len(train_loader)}")
    # print(f"    Test batches: {len(test_loader)}")

    return train_loader, test_loader

# ============================= MODEL SET-UP =================================


def model_set(lrate=LEARNING_RATE) -> tuple[LeNet5, nn.CrossEntropyLoss, optim.Adam]:
    model = LeNet5(num_classes=NUM_CLASSES).to(DEVICE)
    # print(f" Model LeNet5 on: {DEVICE}")
    # print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lrate)
    criterion = nn.CrossEntropyLoss()

    # print(f"    Loss function: CrossEntropyLoss")
    # print(f"     Optimizer LR:{lrate}")

    return model, criterion, optimizer


def train_epoch(model, loader, criterion, optimizer, device, epoch, writer, step=STEP, log_images=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc='  Training', leave=False, position=1)
    for batch_idx, (images, labels) in enumerate(loop):

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        running_acc = float(correct) / float(total)
        # Update progress bar
        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

        # Tensorboard -> per batch (co 10 batchy)
        if batch_idx % 10 == 0:
            step += 1
            writer.add_scalar('batch/training-loss', loss.item(), step)
            writer.add_scalar('batch/training-accuracy', running_acc, step)

            # Loguj obrazki TYLKO dla pierwszego LR w batch_size i tylko pierwszy batch
            # Pokazuj WSZYSTKIE obrazki z pierwszego batcha (batch_size = ilość obrazków)
            if log_images and batch_idx == 0:
                # Dynamiczny nrow - więcej kolumn dla większych batchy
                if images.size(0) <= 64:
                    nrow = 8  # Dla małych batchy: 8 kolumn
                elif images.size(0) <= 256:
                    nrow = 16  # Dla średnich: 16 kolumn
                else:
                    nrow = 32  # Dla dużych (1024, 2048, 4096): 32 kolumny

                # Wszystkie obrazki z batcha
                img_grid = make_grid(
                    images, nrow=nrow, normalize=True, scale_each=True, padding=2)
                writer.add_image("samples/FMNIST_images", img_grid, step)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc, step  # ZWRÓĆ step!


def evaluate(model, loader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc='  Evaluating', leave=False, position=1)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def main() -> int:
    import os
    os.makedirs(FIGURES_PATH, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # Hyperparametry
    batch_sizes = [8192, 4096, 2048, 1024, 512, 256, 64, 32, 16, 8]
    learning_rates = [10**(-np.random.rand() * 4) for _ in range(10)]

    global_step = 0

    for bsize_idx, bsize in enumerate(batch_sizes, 1):
        train_loader, test_loader = dataset_load(bsize=bsize)

        # Flaga - obrazki tylko dla pierwszego LR w każdym batch_size
        first_lr_for_bsize = True

        for lr_idx, lr in enumerate(learning_rates, 1):
            combo_num = (bsize_idx - 1) * len(learning_rates) + lr_idx
            total_combos = len(batch_sizes) * len(learning_rates)
            writer = SummaryWriter(
                TENSORBOARD_LOG_DIR+f"/batch_{bsize}_lr_{lr:.4e}")
            model, criterion, optimizer = model_set(lrate=lr)

            # LISTY PER KOMBINACJA - NIE GLOBALNE!
            run_train_losses = []
            run_train_accs = []
            run_test_losses = []
            run_test_accs = []

            # Progress bar dla epok w tej kombinacji
            epoch_pbar = tqdm(range(EPOCHS),
                              desc=f"[Combo {combo_num}/{total_combos}] Batch={bsize}, LR={lr:.2e}",
                              leave=True,
                              position=0)

            for epoch in epoch_pbar:

                # ZWRÓĆ step!
                train_loss, train_acc, global_step = train_epoch(
                    model, train_loader, criterion, optimizer, DEVICE, epoch, writer,
                    global_step, log_images=first_lr_for_bsize)  # Obrazki tylko dla pierwszego LR!

                test_loss, test_acc = evaluate(
                    model, test_loader, criterion, DEVICE, epoch, writer)

                # Dodaj do list PER KOMBINACJA
                run_train_losses.append(train_loss)
                run_train_accs.append(train_acc)
                run_test_losses.append(test_loss)
                run_test_accs.append(test_acc)

                # Update progress bar with current metrics
                epoch_pbar.set_postfix({
                    'epoch': f'{epoch+1}/{EPOCHS}',
                    'train_acc': f'{train_acc:.2f}%',
                    'test_acc': f'{test_acc:.2f}%',
                    'test_loss': f'{test_loss:.4f}'
                })

                # Logowanie per epoka
                writer.add_scalars('epoch/loss', {
                    'train': train_loss,
                    'test': test_loss
                }, epoch)

                writer.add_scalars('epoch/accuracy', {
                    'train': train_acc,
                    'test': test_acc
                }, epoch)

                writer.add_scalar('hyperparams/learning_rate', lr, epoch)
                writer.add_scalar('hyperparams/batch_size', bsize, epoch)

                # Confusion Matrix dla KAŻDEJ epoki (przewijalne w TensorBoard)
                plot_confusion_matrix(
                    model, test_loader, DEVICE, class_names,
                    writer, epoch, tag=f'confusion_matrix/batch_{bsize}_lr_{lr:.4e}'
                )

            # add_hparams TYLKO RAZ po wszystkich epokach dla TEJ kombinacji!
            writer.add_hparams(
                {
                    'batch_size': bsize,
                    'learning_rate': lr,
                    'epochs': EPOCHS
                },
                {
                    'hparam/train/final_loss': run_train_losses[-1],
                    'hparam/train/final_acc': run_train_accs[-1],
                    'hparam/test/final_loss': run_test_losses[-1],
                    'hparam/test/final_acc': run_test_accs[-1],
                    'hparam/test/best_acc': max(run_test_accs),
                    'hparam/test/worst_acc': min(run_test_accs),
                    'hparam/train/best_acc': max(run_train_accs),
                    'hparam/train/worst_acc': min(run_train_accs),
                    'hparam/test/best_loss': min(run_test_losses),
                    'hparam/test/worst_loss': max(run_test_losses),
                }
            )

            # Po pierwszym LR dla tego batch_size, wyłącz logowanie obrazków
            first_lr_for_bsize = False

            writer.close()

    print("Training finished!")
    # torch.save(model.state_dict(), 'lenet5_fashion_mnist.pth')
    return 0


if __name__ == "__main__":
    main()
