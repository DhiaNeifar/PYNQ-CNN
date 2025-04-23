import numpy as np

def train(model, x_train, y_train, x_test, y_test, epochs=5, lr=0.01):
    for epoch in range(epochs):
        loss = model.forward(x_train, y_train)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        preds = model.predict(x_test)
        acc = np.mean(preds == np.argmax(y_test, axis=1))
        print(f"Test Accuracy: {acc * 100:.2f}%")
