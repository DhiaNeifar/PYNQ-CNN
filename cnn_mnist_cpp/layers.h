#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

using Tensor4D = std::vector<std::vector<std::vector<std::vector<double>>>>; // Use double
using Matrix = std::vector<std::vector<double>>; // Use double

double randn(double stddev) { // Use double
    static std::mt19937 gen(std::random_device{}());
    static std::normal_distribution<double> dist(0.0, 1.0); // Use double
    return dist(gen) * stddev;
}

// ───────────────────────────
// Conv2D
class Conv2D {
public:
    int in_channels, out_channels, kernel_size;
    std::vector<std::vector<std::vector<std::vector<double>>>> weights; // Use double
    std::vector<double> biases; // Use double

    Tensor4D input;

    Conv2D(int in_ch, int out_ch, int k)
        : in_channels(in_ch), out_channels(out_ch), kernel_size(k) {
        double stddev = std::sqrt(2.0 / (in_ch * k * k)); // Use double
        weights.resize(out_ch, std::vector<std::vector<std::vector<double>>>( // Use double
                                  in_ch, std::vector<std::vector<double>>( // Use double
                                             k, std::vector<double>(k)))); // Use double
        biases.resize(out_ch, 0.0); // Use double

        for (int o = 0; o < out_ch; ++o)
            for (int i = 0; i < in_ch; ++i)
                for (int x = 0; x < k; ++x)
                    for (int y = 0; y < k; ++y)
                        weights[o][i][x][y] = randn(stddev);
    }

    Tensor4D forward(const Tensor4D& x) {
        input = x;
        int batch = x.size();
        int h = x[0][0].size();
        int w = x[0][0][0].size();
        int out_h = h - kernel_size + 1;
        int out_w = w - kernel_size + 1;

        Tensor4D output(batch,
                        std::vector<std::vector<std::vector<double>>>( // Use double
                            out_channels,
                            std::vector<std::vector<double>>(out_h, std::vector<double>(out_w, 0.0)))); // Use double

        for (int b = 0; b < batch; ++b) {
            for (int o = 0; o < out_channels; ++o) {
                for (int i = 0; i < out_h; ++i) {
                    for (int j = 0; j < out_w; ++j) {
                        double sum = 0.0; // Use double
                        for (int c = 0; c < in_channels; ++c) {
                            for (int m = 0; m < kernel_size; ++m)
                                for (int n = 0; n < kernel_size; ++n)
                                    sum += x[b][c][i + m][j + n] * weights[o][c][m][n];
                        }
                        output[b][o][i][j] = sum + biases[o];
                    }
                }
            }
        }
        return output;
    }

    Tensor4D backward(const Tensor4D& d_out, double lr) { // Use double
        int batch = input.size();
        int in_h = input[0][0].size();
        int in_w = input[0][0][0].size();
        int out_h = d_out[0][0].size();
        int out_w = d_out[0][0][0].size();

        Tensor4D d_input(batch,
                         std::vector<std::vector<std::vector<double>>>( // Use double
                             in_channels,
                             std::vector<std::vector<double>>(in_h, std::vector<double>(in_w, 0.0)))); // Use double

        auto dw = weights; //copy
        for (auto& f : dw)
            for (auto& c : f)
                for (auto& r : c)
                    std::fill(r.begin(), r.end(), 0.0); // Use double

        std::vector<double> db(out_channels, 0.0); // Use double

        for (int b = 0; b < batch; ++b) {
            for (int o = 0; o < out_channels; ++o) {
                for (int i = 0; i < out_h; ++i) {
                    for (int j = 0; j < out_w; ++j) {
                        double grad = d_out[b][o][i][j]; // Use double
                        db[o] += grad;
                        for (int c = 0; c < in_channels; ++c) {
                            for (int m = 0; m < kernel_size; ++m)
                                for (int n = 0; n < kernel_size; ++n) {
                                    dw[o][c][m][n] += grad * input[b][c][i + m][j + n];
                                    d_input[b][c][i + m][j + n] += grad * weights[o][c][m][n];
                                }
                        }
                    }
                }
            }
        }

        for (int o = 0; o < out_channels; ++o) {
            for (int c = 0; c < in_channels; ++c)
                for (int m = 0; m < kernel_size; ++m)
                    for (int n = 0; n < kernel_size; ++n)
                        weights[o][c][m][n] -= lr * dw[o][c][m][n];
            biases[o] -= lr * db[o];
        }

        return d_input;
    }
};

// ───────────────────────────
// ReLU
class ReLU {
public:
    Tensor4D mask;

    Tensor4D forward(const Tensor4D& x) {
        mask = x;
        Tensor4D out = x;

        int active_count = 0;
        int total_count = 0;

        for (int b = 0; b < x.size(); ++b)
            for (int c = 0; c < x[0].size(); ++c)
                for (int i = 0; i < x[0][0].size(); ++i)
                    for (int j = 0; j < x[0][0][0].size(); ++j) {
                        double val = x[b][c][i][j]; // Use double
                        if (val > 0.0) { // Changed from 0.0f to 0.0
                            mask[b][c][i][j] = 1.0; // Use double
                            ++active_count;
                        } else {
                            mask[b][c][i][j] = 0.0; // Use double
                            out[b][c][i][j] = 0.0; // Use double
                        }
                        ++total_count;
                    }

        return out;
    }

    Tensor4D backward(const Tensor4D& d_out, double) { // Use double
        Tensor4D out = d_out;
        for (int b = 0; b < d_out.size(); ++b)
            for (int c = 0; c < d_out[0].size(); ++c)
                for (int i = 0; i < d_out[0][0].size(); ++i)
                    for (int j = 0; j < d_out[0][0][0].size(); ++j)
                        out[b][c][i][j] *= mask[b][c][i][j];
        return out;
    }
};

class ReLU2D {
public:
    Matrix mask;

    Matrix forward(const Matrix& x) {
        int batch = x.size();
        int features = x[0].size();

        mask = Matrix(batch, std::vector<double>(features));
        Matrix out = x;

        int active_count = 0;
        int total_count = batch * features;

        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < features; ++j) {
                if (x[i][j] > 0.0) {
                    mask[i][j] = 1.0;
                    ++active_count;
                } else {
                    mask[i][j] = 0.0;
                    out[i][j] = 0.0;
                }
            }
        }

        return out;
    }

    Matrix backward(const Matrix& d_out, double) {
        int batch = d_out.size();
        int features = d_out[0].size();

        Matrix grad = d_out;

        for (int i = 0; i < batch; ++i)
            for (int j = 0; j < features; ++j)
                grad[i][j] *= mask[i][j];

        return grad;
    }
};

// ───────────────────────────
// MaxPool2D
class MaxPool2D {
public:
    Tensor4D input;
    Tensor4D mask;
    int pool_size = 2;

    Tensor4D forward(const Tensor4D& x) {
        input = x;
        int batch = x.size();
        int channels = x[0].size();
        int h = x[0][0].size();
        int w = x[0][0][0].size();
        int out_h = h / pool_size;
        int out_w = w / pool_size;

        Tensor4D out(batch,
                     std::vector<std::vector<std::vector<double>>>(
                         channels,
                         std::vector<std::vector<double>>(
                             out_h, std::vector<double>(out_w))));

        // Initialize the mask with zeros
        mask = Tensor4D(batch,
                        std::vector<std::vector<std::vector<double>>>(
                            channels,
                            std::vector<std::vector<double>>(h, std::vector<double>(w, 0.0))));

        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < out_h; ++i) {
                    for (int j = 0; j < out_w; ++j) {
                        double max_val = -1e9;
                        int max_i = -1, max_j = -1;

                        for (int m = 0; m < pool_size; ++m) {
                            for (int n = 0; n < pool_size; ++n) {
                                int row = i * pool_size + m;
                                int col = j * pool_size + n;
                                double val = x[b][c][row][col];
                                if (val > max_val) {
                                    max_val = val;
                                    max_i = row;
                                    max_j = col;
                                }
                            }
                        }

                        out[b][c][i][j] = max_val;
                        mask[b][c][max_i][max_j] = 1.0;
                    }
                }
            }
        }

        return out;
    }

    Tensor4D backward(const Tensor4D& d_out, double) {
        int batch = d_out.size();
        int channels = d_out[0].size();
        int out_h = d_out[0][0].size();
        int out_w = d_out[0][0][0].size();
        int h = out_h * pool_size;
        int w = out_w * pool_size;

        Tensor4D d_input(batch,
                         std::vector<std::vector<std::vector<double>>>(
                             channels,
                             std::vector<std::vector<double>>(h, std::vector<double>(w, 0.0))));

        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < out_h; ++i) {
                    for (int j = 0; j < out_w; ++j) {
                        for (int m = 0; m < pool_size; ++m) {
                            for (int n = 0; n < pool_size; ++n) {
                                int row = i * pool_size + m;
                                int col = j * pool_size + n;
                                if (mask[b][c][row][col] == 1.0)
                                    d_input[b][c][row][col] = d_out[b][c][i][j];
                            }
                        }
                    }
                }
            }
        }

        return d_input;
    }
};


// ───────────────────────────
// Flatten
class Flatten {
public:
    int batch, channels, height, width;

    std::vector<std::vector<double>> forward(const Tensor4D& x) { // Use double
        batch = x.size();
        channels = x[0].size();
        height = x[0][0].size();
        width = x[0][0][0].size();
        std::vector<std::vector<double>> out(batch, std::vector<double>(channels * height * width)); // Use double
        for (int b = 0; b < batch; ++b) {
            int idx = 0;
            for (int c = 0; c < channels; ++c)
                for (int i = 0; i < height; ++i)
                    for (int j = 0; j < width; ++j)
                        out[b][idx++] = x[b][c][i][j];
        }
        return out;
    }

    Tensor4D backward(const std::vector<std::vector<double>>& d_out) { // Use double
        Tensor4D out(batch,
                     std::vector<std::vector<std::vector<double>>>( // Use double
                         channels,
                         std::vector<std::vector<double>>(
                             height, std::vector<double>(width)))); // Use double
        for (int b = 0; b < batch; ++b) {
            int idx = 0;
            for (int c = 0; c < channels; ++c)
                for (int i = 0; i < height; ++i)
                    for (int j = 0; j < width; ++j)
                        out[b][c][i][j] = d_out[b][idx++];
        }
        return out;
    }
};

// ───────────────────────────
// Dense
class Dense {
public:
    Matrix weights;
    std::vector<double> biases;
    Matrix input;

    Dense(int in_features, int out_features) {
        weights.resize(in_features, std::vector<double>(out_features)); // Use double
        biases.resize(out_features, 0.0); // Use double
        double stddev = std::sqrt(2.0 / in_features); // Use double
        for (auto& row : weights)
            for (auto& val : row)
                val = randn(stddev);
    }

    Matrix forward(const Matrix& x) {
        input = x;
        int batch = x.size();
        int out_dim = biases.size();
        Matrix out(batch, std::vector<double>(out_dim, 0.0)); // Use double
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < out_dim; ++j) {
                out[b][j] = biases[j];
                for (int i = 0; i < x[0].size(); ++i)
                    out[b][j] += x[b][i] * weights[i][j];
            }
        return out;
    }

    Matrix backward(const Matrix& d_out, double lr) { // Use double
        int batch = d_out.size();
        int in_dim = weights.size();
        int out_dim = weights[0].size();

        Matrix d_input(batch, std::vector<double>(in_dim, 0.0)); // Use double
        Matrix d_weights(in_dim, std::vector<double>(out_dim, 0.0)); // Use double
        std::vector<double> d_biases(out_dim, 0.0); // Use double

        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < out_dim; ++j) {
                d_biases[j] += d_out[b][j];
                for (int i = 0; i < in_dim; ++i) {
                    d_weights[i][j] += input[b][i] * d_out[b][j];
                    d_input[b][i] += d_out[b][j] * weights[i][j];
                }
            }
        }
        // ✅ Actually update model weights and biases
        for (int i = 0; i < in_dim; ++i)
            for (int j = 0; j < out_dim; ++j)
                weights[i][j] -= lr * d_weights[i][j];  // modify actual member

        for (int j = 0; j < out_dim; ++j)
            biases[j] -= lr * d_biases[j];
        return d_input;
    }
};

#endif
