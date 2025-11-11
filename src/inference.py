from hparams import *
from dataloader import *
from model import *
import random

import time
# ====================================
import matplotlib.pyplot as plt


def plot_inference(images=None, labels=None, figtitle='Fashion_MNIST_labels', predicted_output=None):

    label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                   "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    if images is None or labels is None:
        _, dataloader = dataset_load()
        images, labels = next(iter(dataloader))

    fig, axes = plt.subplots(2, 4, figsize=(4, 4), gridspec_kw={
                             'wspace': 0, 'hspace': 0})
    fig.canvas.manager.set_window_title(figtitle)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            break

        image = images[idx].squeeze()
        label = labels[idx].item()
        ax.imshow(image, cmap='gray', aspect='auto')

        if predicted_output is not None:
            predicted_label = predicted_output[idx].item()
            is_correct = (label == predicted_label)
            if is_correct:
                title_text = label_names[label]
                text_color = 'lime'
            else:
                title_text = f"{label_names[label]}[{label_names[predicted_label]}]"
                text_color = 'red'
        else:
            title_text = label_names[label]
            text_color = 'white'
        ax.text(0.5, 0.95, title_text,
                color=text_color, fontsize=11, weight='bold',
                ha='center', va='top', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for letter in [r' ', r'\n', r"\t", r",", r"/", r"-", r"\\"]:
        figtitle = figtitle.replace(letter, '_')

    plt.savefig("../visuals/" + figtitle + ".png")
    plt.show()


def main() -> int:
    # Detect if model is quantized by checking filename
    is_quantized = 'int8' in MODEL_PATH or 'quantized' in MODEL_PATH or 'int4' in MODEL_PATH

    if is_quantized:
        print(f"⚙️  Loading quantized model: {MODEL_PATH}")
        model = model_load_quantized(MODEL_PATH, device='cpu')
        device = 'cpu'  # Quantized models only work on CPU
    else:
        print(f"⚙️  Loading float32 model: {MODEL_PATH}")
        model = model_load(MODEL_PATH)
        device = DEVICE

    model.eval()

    _, dataloader = dataset_load()

    batch_idx = random.randint(0, len(dataloader) - 1)
    for idx, (images, labels) in enumerate(dataloader):
        if idx == batch_idx:
            break

    print(f"Selected batch: {batch_idx + 1}/{len(dataloader)}")
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():

        # Inference na wybranym batchu
        t_start = time.time()

        outputs = model(images)

        t_stop = time.time() - t_start
        t_per_sample = (t_stop / len(outputs)) * 1e3
        t_stop *= 1e3

        predictions = outputs.argmax(dim=1)
        current_accuracy = (predictions == labels).float().mean().item()

        # Try to extract accuracy from filename (if available)
        try:
            mp = MODEL_PATH.strip().split("_")
            acc = mp[-1].split(".pth")
            acc = float(acc[0])
            print(f"  Accuracy (overall): {acc:.4f} [%]")
        except (ValueError, IndexError):
            print(f"  Accuracy (overall): N/A (quantized model)")

        print("        Ground truth:", labels.cpu().numpy())
        print("          Prediction:", predictions.cpu().numpy())
        print(f"  Accuracy (current): {(current_accuracy*100):.4f}  [%]")
        print(f"   Time per batch({BATCH_SIZE}): {t_stop:.4f} [ms]")
        print(f"     Time per sample: {t_per_sample:.4f}  [ms]")

        plot_inference(
            images=images.cpu(),
            labels=labels.cpu(),
            figtitle='Fashion MNIST Predictions',
            predicted_output=predictions.cpu()
        )

    return 0


if __name__ == "__main__":
    main()
