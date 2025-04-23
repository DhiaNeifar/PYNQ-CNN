import numpy as np
import matplotlib.pyplot as plt


def save_model(model, path):
    params = {
        "c1_w": model.c1.weights, "c1_b": model.c1.biases,
        "c2_w": model.c2.weights, "c2_b": model.c2.biases,
        "fc1_w": model.fc1.weights, "fc1_b": model.fc1.biases,
        "fc2_w": model.fc2.weights, "fc2_b": model.fc2.biases
    }
    np.savez(path, **params)


def load_model(model, path):
    params = np.load(path)
    model.c1.weights = params["c1_w"]
    model.c1.biases = params["c1_b"]
    model.c2.weights = params["c2_w"]
    model.c2.biases = params["c2_b"]
    model.fc1.weights = params["fc1_w"]
    model.fc1.biases = params["fc1_b"]
    model.fc2.weights = params["fc2_w"]
    model.fc2.biases = params["fc2_b"]


def plot_metrics(losses, accuracies, title="Training Progress"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, marker='o', linestyle='--', label='Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, marker='s', linestyle='-', label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
