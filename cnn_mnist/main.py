# main.py
import numpy as np
import time
from tqdm import tqdm
from data_loader import load_mnist
from model import CNN
from utils import save_model, plot_metrics

# Load data
x_train, y_train, x_test, y_test = load_mnist()

model = CNN()
lr = 0.01
epochs = 5
batch_size = 64

train_loss, train_acc = [], []

start_time = time.time()  # Total training time

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    epoch_start = time.time()

    perm = np.random.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    losses, correct = [], 0

    for i in tqdm(range(0, len(x_train), batch_size), desc="Training", leave=False):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        loss = model.forward(x_batch, y_batch)
        model.backward(lr)
        losses.append(loss)

        preds = model.predict(x_batch)
        correct += (preds == y_batch).sum()

    acc = correct / len(x_train)
    train_loss.append(np.mean(losses))
    train_acc.append(acc)
    epoch_end = time.time()

    print(f"Loss: {train_loss[-1]:.4f}, Accuracy: {train_acc[-1]*100:.2f}%")
    print(f"Epoch Time: {epoch_end - epoch_start:.2f} seconds")

    # Save model and metrics after each epoch (overwrite)
    save_model(model, "trained_model.npz")
    np.save("train_loss.npy", np.array(train_loss))
    np.save("train_acc.npy", np.array(train_acc))

end_time = time.time()
print(f"\nTotal Training Time: {end_time - start_time:.2f} seconds")

# Evaluate on test set
test_preds = model.predict(x_test)
test_acc = np.mean(test_preds == y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot metrics
plot_metrics(train_loss, train_acc, "Training Metrics")
