from hparams import *
from tqdm import tqdm

import numpy as np
from src.dataloader import *
from model import *

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.hparams_tuning import plot_confusion_matrix


def train_epoch(model, loader, criterion, optimizer, device, epoch, writer, step=STEP, log_images=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_losses = []

    loop = tqdm(loader, desc='Training', leave=False)
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

        batch_losses.append(loss.item())
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        running_acc = float(correct) / float(total)
        # Update progress bar
        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

        # Tensorboard -> per batch
        step += 1
        writer.add_scalar('batch/training-loss', loss.item(), step)
        writer.add_scalar('batch/training-accuracy', running_acc, step)
        #
        if log_images and batch_idx == 0:
            img_grid = make_grid(images[:8])
            writer.add_image(f"FMNIST_Images", img_grid, step)
        #
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc='Evaluating', leave=False)
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


def main():
    # Fashion MNIST class names
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    train_loader, test_loader = dataset_load(print_params=True)
    model, criterion, optimizer = model_set()

    # Train the model
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    best_acc = 0.0
    best_model_path = None

    # Early stopping
    patience = 20
    epochs_without_improvement = 0

    writer = SummaryWriter(
        TENSORBOARD_LOG_DIR+f"/batch_size_{BATCH_SIZE}_lr_{LEARNING_RATE:.4e}")
    print("Training...")

    epoch_loop = tqdm(range(EPOCHS), desc='Epochs', leave=False)
    for epoch in epoch_loop:
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch, writer, log_images=(epoch == 0))

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, DEVICE, epoch, writer)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        writer.add_scalars('epoch/loss', {
            'train': train_loss,
            'test': test_loss
        }, epoch)

        writer.add_scalars('epoch/accuracy', {
            'train': train_acc,
            'test': test_acc
        }, epoch)

        # Save best model and early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = f'models/lenet5_fashion_mnist_{test_acc/100:.4f}.pth'
            torch.save(model.state_dict(), best_model_path)
            epochs_without_improvement = 0  # Reset counter
            epoch_loop.set_description(
                f'Epochs (Best: {best_acc:.2f}% Saved)')
        else:
            epochs_without_improvement += 1
            epoch_loop.set_description(
                f'Epochs (Best: {best_acc:.2f}%, No improvement: {epochs_without_improvement}/{patience})')

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered! No improvement for {patience} epochs.")
            break

    print('-' * 50)
    if epochs_without_improvement >= patience:
        print(f"Training stopped early at epoch {epoch + 1}/{EPOCHS}")
    else:
        print("Training finished.")
    print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")
    print(f" Best Test Accuracy: {best_acc:.2f}%")
    print(f"Best model saved as: {best_model_path}")

    # Generate confusion matrix for best model and log to TensorBoard

    cm = plot_confusion_matrix(model, test_loader, DEVICE, class_names,
                               writer, global_step=EPOCHS-1, tag='final/confusion_matrix')

    writer.close()
    return 0


if __name__ == "__main__":
    main()
