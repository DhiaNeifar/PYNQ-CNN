import numpy as np

from data_loader import load_mnist
from model import CNN
from utils import load_model

model = CNN()
load_model(model, 'trained_model.npz')

# Load data
x_train, y_train, x_test, y_test = load_mnist()

# Evaluate on test set
test_preds = model.predict(x_test)
test_acc = np.mean(test_preds == y_test)
print(f"Test Accuracy: {test_acc:.4f}")
