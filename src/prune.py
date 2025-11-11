import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import time
from tqdm import tqdm

import numpy as np


from model import *
from hparams import *
from dataloader import *
import random
import matplotlib.pyplot as plt
from inference import *


def analyze_model(model, num_batches=400, phase_name=""):
    """Analyze model accuracy on test batches"""
    model.eval()
    _, dataloader = dataset_load()

    # Statistics tracking
    accuracies = []
    total_time = 0.0

    # Class names for Fashion MNIST
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Track misclassifications per class
    class_errors = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}

    print(f"\n{'='*70}")
    print(f"Analyzing {phase_name}model on first {num_batches} batches")
    print(f"{'='*70}\n")

    total_batches = len(dataloader)

    # Calculate print interval: sqrt(num_batches) rounded
    import math
    print_interval = max(1, round(math.sqrt(num_batches)))
    print(f"Printing every {print_interval} batches...\n")

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            # Stop after num_batches
            if idx >= num_batches:
                break

            test_num = idx + 1

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Inference timing
            t_start = time.time()
            outputs = model(images)
            t_stop = time.time() - t_start

            # Calculate metrics
            predictions = outputs.argmax(dim=1)
            batch_accuracy = (predictions == labels).float().mean().item()
            t_per_sample = (t_stop / len(outputs)) * 1e3
            t_stop_ms = t_stop * 1e3

            # Track per-class errors
            for label, pred in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                class_total[label] += 1
                if label != pred:
                    class_errors[label] += 1

            # Accumulate statistics
            accuracies.append(batch_accuracy * 100)
            total_time += t_stop_ms

            # Print batch results at intervals
            if test_num % print_interval == 0 or test_num == num_batches:
                print(f"Batch {test_num}/{num_batches}: Acc={batch_accuracy*100:5.1f}%  Time={t_stop_ms:5.2f}ms  "
                      f"GT={labels.cpu().numpy()}  Pred={predictions.cpu().numpy()}")

    # Calculate statistics
    accuracies_np = np.array(accuracies)
    mean_accuracy = np.mean(accuracies_np)
    std_accuracy = np.std(accuracies_np)
    avg_time = total_time / num_batches
    avg_time_per_sample = avg_time / BATCH_SIZE

    # Calculate per-class error rates
    class_error_rates = []
    for class_id in range(10):
        if class_total[class_id] > 0:
            error_rate = (class_errors[class_id] / class_total[class_id]) * 100
            class_error_rates.append((class_id, class_names[class_id],
                                     error_rate, class_errors[class_id],
                                     class_total[class_id]))

    # Group accuracies into 10% ranges
    accuracy_distribution = {
        "0-10%": 0, "10-20%": 0, "20-30%": 0, "30-40%": 0, "40-50%": 0,
        "50-60%": 0, "60-70%": 0, "70-80%": 0, "80-90%": 0, "90-100%": 0
    }

    for acc in accuracies:
        if acc < 10:
            accuracy_distribution["0-10%"] += 1
        elif acc < 20:
            accuracy_distribution["10-20%"] += 1
        elif acc < 30:
            accuracy_distribution["20-30%"] += 1
        elif acc < 40:
            accuracy_distribution["30-40%"] += 1
        elif acc < 50:
            accuracy_distribution["40-50%"] += 1
        elif acc < 60:
            accuracy_distribution["50-60%"] += 1
        elif acc < 70:
            accuracy_distribution["60-70%"] += 1
        elif acc < 80:
            accuracy_distribution["70-80%"] += 1
        elif acc < 90:
            accuracy_distribution["80-90%"] += 1
        else:
            accuracy_distribution["90-100%"] += 1

    print(f"{'='*70}")
    print(f"SUMMARY STATISTICS - {phase_name}")
    print(f"{'='*70}")
    print("Accuracy Distribution:")
    for range_name, count in accuracy_distribution.items():
        # Scale down for readability
        bar = "█" * (count // 5 if count > 0 else 0)
        print(f"  {range_name:9s}: {count:3d} batches {bar}")

    print(f"\nMean Accuracy:          {mean_accuracy:.2f}%")
    print(f"Std Deviation:          {std_accuracy:.2f}%")
    print(f"Average Time (batch):   {avg_time:.2f} ms")
    print(f"Average Time (sample):  {avg_time_per_sample:.2f} ms")

    print(f"\n{'='*70}")
    print(f"MOST MISCLASSIFIED CLASSES - {phase_name}")
    print(f"{'='*70}")
    for class_id, class_name, error_rate, errors, total in class_error_rates:
        if total > 0:
            print(
                f"{class_id}. {class_name:15s}: {error_rate:5.1f}% error ({errors}/{total} wrong)")

    print(f"{'='*70}\n")

    return mean_accuracy


def main() -> int:
    print("\n" + "="*70)
    print("PHASE 1: PRUNING")
    print(f"Using device: {DEVICE}")
    print("="*70)

    model = model_load("prune/lenet5_fashion_mnist_0.9127.pth")
    model.to(DEVICE)
    print(f"Model loaded on: {next(model.parameters()).device}\n")

    prune.ln_structured(model.conv1, name="weight",
                        amount=0.15, n=2, dim=0)
    prune.ln_structured(model.conv2, name="weight",
                        amount=0.15, n=2, dim=0)
    prune.ln_structured(model.fc1, name="weight",
                        amount=0.15, n=2, dim=0)
    prune.ln_structured(model.fc2, name="weight",
                        amount=0.30, n=2, dim=0)

    # Make pruning permanent (remove masks and actually delete weights)
    print("Making pruning permanent (removing masks)...")
    prune.remove(model.conv1, 'weight')
    prune.remove(model.conv2, 'weight')
    prune.remove(model.fc1, 'weight')
    prune.remove(model.fc2, 'weight')
    print("✓ Pruning made permanent - weights actually removed\n")

    # Show pruning statistics
    print("-"*70)
    print("PRUNING STATISTICS")
    print("-"*70)

    total_params = 0
    zero_params = 0

    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            weight = module.weight
            total = weight.numel()
            zeros = (weight == 0).sum().item()

            total_params += total
            zero_params += zeros

            if zeros > 0:
                print(
                    f"{name:10s}: {total:6d} params, {zeros:6d} zeros ({100*zeros/total:5.1f}%)")

    print("-"*70)
    print(
        f"TOTAL:      {total_params:6d} params, {zero_params:6d} zeros ({100*zero_params/total_params:5.1f}%)")
    print(
        f"Effective compression: {100*zero_params/total_params:.1f}% params are zero")
    print(f"NOTE: Model file size stays same - zeros are still stored!")
    print("-"*70 + "\n")

    # Analyze after pruning - FULL 1250 BATCHES
    accuracy_before = analyze_model(
        model, num_batches=1250, phase_name="AFTER PRUNING ")

    # PHASE 2: FINE-TUNING ON CUDA
    print("\n" + "="*70)
    print(f"PHASE 2: FINE-TUNING (10 EPOCHS) ON {DEVICE}")
    print("="*70 + "\n")

    # Make sure model is on CUDA for training
    model.to(DEVICE)
    print(f"Model is on: {next(model.parameters()).device}")

    train_loader, test_loader = dataset_load()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/10', leave=True)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Track accuracy during training
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = epoch_loss / len(train_loader)

        # Quick accuracy check on test set every epoch
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        train_acc = 100 * correct / total
        print(
            f"Epoch {epoch+1}/10 Summary: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    # Save finetuned model
    torch.save(model.state_dict(), "lenet5_pruned_finetuned.pth")
    print(f"\n✓ Model saved to: lenet5_pruned_finetuned.pth")
    print(f"Model is still on: {next(model.parameters()).device}\n")

    # PHASE 3: ANALYZE AFTER FINE-TUNING - FULL 1250 BATCHES
    print("\n" + "="*70)
    print("PHASE 3: FINAL ANALYSIS")
    print("="*70)

    accuracy_after = analyze_model(
        model, num_batches=1250, phase_name="AFTER FINE-TUNING ")

    # Final comparison
    print("\n" + "="*70)
    print("IMPROVEMENT SUMMARY")
    print("="*70)
    print(f"Before Fine-tuning: {accuracy_before:.2f}%")
    print(f"After Fine-tuning:  {accuracy_after:.2f}%")
    print(f"Improvement:        +{accuracy_after - accuracy_before:.2f}%")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    main()
