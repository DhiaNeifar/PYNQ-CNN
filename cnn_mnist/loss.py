import numpy as np


class SoftmaxCrossEntropy:
    def forward(self, x, y):
        self.y = y
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        return -np.mean(np.log(self.probs[range(len(y)), y] + 1e-9))

    def backward(self):
        grad = self.probs
        grad[range(len(self.y)), self.y] -= 1
        return grad / len(self.y)
