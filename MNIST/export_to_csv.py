import gzip
import pickle
import numpy as np
import csv
import collections

def load_balanced_mnist(path="../MNIST/mnist.pkl.gz", train_per_class=500, test_per_class=10):
    with gzip.open(path, "rb") as f:
        train_set, val_set, test_set = pickle.load(f, encoding="latin1")

    x_train_full = np.concatenate([train_set[0], val_set[0]], axis=0)
    y_train_full = np.concatenate([train_set[1], val_set[1]], axis=0)
    x_test_full = test_set[0]
    y_test_full = test_set[1]

    x_train_bal, y_train_bal = [], []
    counter_train = collections.defaultdict(int)

    for x, y in zip(x_train_full, y_train_full):
        if counter_train[y] < train_per_class:
            x_train_bal.append(x)
            y_train_bal.append(y)
            counter_train[y] += 1
        if all(v >= train_per_class for v in counter_train.values()):
            break

    x_test_bal, y_test_bal = [], []
    counter_test = collections.defaultdict(int)

    for x, y in zip(x_test_full, y_test_full):
        if counter_test[y] < test_per_class:
            x_test_bal.append(x)
            y_test_bal.append(y)
            counter_test[y] += 1
        if all(v >= test_per_class for v in counter_test.values()):
            break

    # Do NOT normalize
    x_train = np.array(x_train_bal).astype(np.float32)
    y_train = np.array(y_train_bal)
    x_test = np.array(x_test_bal).astype(np.float32)
    y_test = np.array(y_test_bal)

    return x_train, y_train, x_test, y_test

def save_to_csv(filename, array):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for row in array:
            writer.writerow(row)

def main():
    x_train, y_train, x_test, y_test = load_balanced_mnist()

    save_to_csv("train_images.csv", x_train.reshape(len(x_train), -1))
    save_to_csv("train_labels.csv", y_train.reshape(-1, 1))
    save_to_csv("test_images.csv", x_test.reshape(len(x_test), -1))
    save_to_csv("test_labels.csv", y_test.reshape(-1, 1))

    print("✅ Exported MNIST to CSV without normalization")

if __name__ == "__main__":
    main()
