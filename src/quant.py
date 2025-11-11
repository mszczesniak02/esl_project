import torch
from torch import nn
import torch.quantization
import os
import copy
import time

from model import *
from hparams import *
from src.dataloader import *


def measure_model(model, dataloader, num_batches=400):
    """Measure accuracy and speed"""
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            if idx >= num_batches:
                break

            images, labels = images.to('cpu'), labels.to('cpu')
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_time_ms = ((time.time() - start_time) / num_batches) * 1000
    return accuracy, avg_time_ms


def main() -> int:
    print("\n" + "="*70)
    print("QUANTIZATION: float32 → int8")
    print("="*70 + "\n")

    # Load model and data
    model_fp32 = model_load("prune/lenet5_fashion_mnist_0.9127.pth")
    model_fp32.to('cpu').eval()
    _, test_loader = dataset_load()

    # Baseline
    torch.save(model_fp32.state_dict(), "/tmp/fp32.pth")
    fp32_size = os.path.getsize("/tmp/fp32.pth") / 1024 / 1024
    fp32_acc, fp32_time = measure_model(model_fp32, test_loader)

    # Quantize to int8
    model_int8 = torch.quantization.quantize_dynamic(
        copy.deepcopy(model_fp32),
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )

    # Save quantized model
    torch.save(model_int8.state_dict(), "lenet5_quantized_int8.pth")
    print(f"✓ Quantized model saved to: lenet5_quantized_int8.pth\n")

    torch.save(model_int8.state_dict(), "/tmp/int8.pth")
    int8_size = os.path.getsize("/tmp/int8.pth") / 1024 / 1024
    int8_acc, int8_time = measure_model(model_int8, test_loader)

    # Results
    print(f"{'Model':<12} {'Size (MB)':<12} {'Accuracy':<12} {'Speed (ms)':<12}")
    print("-"*70)
    print(f"{'float32':<12} {fp32_size:>8.2f}    {fp32_acc:>8.2f}%   {fp32_time:>8.2f}")
    print(f"{'int8':<12} {int8_size:>8.2f}    {int8_acc:>8.2f}%   {int8_time:>8.2f}")
    print("-"*70)
    print(f"Compression: {fp32_size/int8_size:.1f}x smaller")
    print(f"Speedup:     {fp32_time/int8_time:.1f}x faster")
    print(f"Accuracy:    {int8_acc - fp32_acc:+.2f}%\n")

    return 0


if __name__ == "__main__":
    main()
