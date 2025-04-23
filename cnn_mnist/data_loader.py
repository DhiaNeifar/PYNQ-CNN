import gzip
import pickle
import numpy as np


def load_mnist(path="../MNIST/mnist.pkl.gz", train_per_class=500, test_per_class=10):
    import collections

    with gzip.open(path, "rb") as f:
        train_set, val_set, test_set = pickle.load(f, encoding="latin1")

    # Combine training + validation
    x_train_full = np.concatenate([train_set[0], val_set[0]], axis=0)
    y_train_full = np.concatenate([train_set[1], val_set[1]], axis=0)
    x_test_full = test_set[0]
    y_test_full = test_set[1]

    # Collect balanced samples for training
    x_train_balanced, y_train_balanced = [], []
    counter_train = collections.defaultdict(int)

    for x, y in zip(x_train_full, y_train_full):
        if counter_train[y] < train_per_class:
            x_train_balanced.append(x)
            y_train_balanced.append(y)
            counter_train[y] += 1
        if all(v >= train_per_class for v in counter_train.values()):
            break

    # Collect balanced samples for testing
    x_test_balanced, y_test_balanced = [], []
    counter_test = collections.defaultdict(int)

    for x, y in zip(x_test_full, y_test_full):
        if counter_test[y] < test_per_class:
            x_test_balanced.append(x)
            y_test_balanced.append(y)
            counter_test[y] += 1
        if all(v >= test_per_class for v in counter_test.values()):
            break

    # Reshape and convert
    x_train = np.array(x_train_balanced).reshape(-1, 1, 28, 28).astype(np.float32)
    y_train = np.array(y_train_balanced)
    x_test = np.array(x_test_balanced).reshape(-1, 1, 28, 28).astype(np.float32)
    y_test = np.array(y_test_balanced)

    return x_train, y_train, x_test, y_test

