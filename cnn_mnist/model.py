from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from loss import SoftmaxCrossEntropy


class CNN:
    def __init__(self):
        self.c1 = Conv2D(1, 32, 3)
        self.r1 = ReLU()
        self.p1 = MaxPool2D()
        self.c2 = Conv2D(32, 64, 3)
        self.r2 = ReLU()
        self.p2 = MaxPool2D()
        self.flat = Flatten()
        self.fc1 = Dense(1600, 128)
        self.r3 = ReLU()
        self.fc2 = Dense(128, 10)
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x, y):
        x = self.c1.forward(x)
        x = self.r1.forward(x)
        x = self.p1.forward(x)
        x = self.c2.forward(x)
        x = self.r2.forward(x)
        x = self.p2.forward(x)
        x = self.flat.forward(x)
        x = self.fc1.forward(x)
        x = self.r3.forward(x)
        x = self.fc2.forward(x)
        self.logits = x
        return self.loss_fn.forward(x, y)

    def backward(self, lr):
        grad = self.loss_fn.backward()
        grad = self.fc2.backward(grad, lr)
        grad = self.r3.backward(grad, lr)
        grad = self.fc1.backward(grad, lr)
        grad = self.flat.backward(grad, lr)
        grad = self.p2.backward(grad, lr)
        grad = self.r2.backward(grad, lr)
        grad = self.c2.backward(grad, lr)
        grad = self.p1.backward(grad, lr)
        grad = self.r1.backward(grad, lr)
        self.c1.backward(grad, lr)

    def predict(self, x):
        x = self.c1.forward(x)
        x = self.r1.forward(x)
        x = self.p1.forward(x)
        x = self.c2.forward(x)
        x = self.r2.forward(x)
        x = self.p2.forward(x)
        x = self.flat.forward(x)
        x = self.fc1.forward(x)
        x = self.r3.forward(x)
        x = self.fc2.forward(x)
        return x.argmax(axis=1)
