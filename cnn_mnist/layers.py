import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.k = kernel_size
        self.weights = np.random.randn(out_channels, in_channels, self.k, self.k) * np.sqrt(
            2. / (in_channels * self.k * self.k))
        self.biases = np.zeros((out_channels, 1))

    def forward(self, x):
        self.x = x
        batch, _, h, w = x.shape
        out_h = h - self.k + 1
        out_w = w - self.k + 1
        self.out = np.zeros((batch, self.weights.shape[0], out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                region = x[:, :, i:i + self.k, j:j + self.k]
                self.out[:, :, i, j] = np.tensordot(region, self.weights, axes=([1, 2, 3], [1, 2, 3])) + self.biases.T
        return self.out

    def backward(self, d_out, lr):
        batch, out_ch, out_h, out_w = d_out.shape
        dx = np.zeros_like(self.x)
        dw = np.zeros_like(self.weights)
        db = np.sum(d_out, axis=(0, 2, 3)).reshape(self.biases.shape)

        for i in range(out_h):
            for j in range(out_w):
                region = self.x[:, :, i:i + self.k, j:j + self.k]
                for b in range(batch):
                    for o in range(out_ch):
                        dw[o] += d_out[b, o, i, j] * region[b]
                        dx[b, :, i:i + self.k, j:j + self.k] += d_out[b, o, i, j] * self.weights[o]

        self.weights -= lr * dw
        self.biases -= lr * db
        return dx


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, d_out, lr):
        return d_out * self.mask


class MaxPool2D:
    def forward(self, x):
        self.x = x
        self.b, self.c, self.h, self.w = x.shape

        self.h_trunc = self.h - (self.h % 2)
        self.w_trunc = self.w - (self.w % 2)
        x = x[:, :, :self.h_trunc, :self.w_trunc]

        self.x_reshaped = x.reshape(self.b, self.c, self.h_trunc // 2, 2, self.w_trunc // 2, 2)
        self.out = np.max(self.x_reshaped, axis=(3, 5), keepdims=True)
        self.mask = (self.x_reshaped == self.out)

        return self.out.squeeze()

    def backward(self, d_out, lr):
        d_out_expanded = d_out[:, :, :, None, :, None]  # shape for broadcasting
        d_input_reshaped = self.mask * d_out_expanded
        d_input = np.zeros_like(self.x[:, :, :self.h_trunc, :self.w_trunc])
        d_input[:, :, :, :] = d_input_reshaped.reshape(self.b, self.c, self.h_trunc, self.w_trunc)

        # pad back to original input shape if needed
        result = np.zeros_like(self.x)
        result[:, :, :self.h_trunc, :self.w_trunc] = d_input
        return result


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out, lr):
        return d_out.reshape(self.shape)


class Dense:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.biases = np.zeros((1, out_features))

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.biases

    def backward(self, d_out, lr):
        dw = self.x.T @ d_out
        db = np.sum(d_out, axis=0, keepdims=True)
        dx = d_out @ self.weights.T
        self.weights -= lr * dw
        self.biases -= lr * db
        return dx
