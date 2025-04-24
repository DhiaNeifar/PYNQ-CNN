#include "data_loader.h"
#include "model.h"
#include "utils.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>

void print_progress_bar(size_t current, size_t total, int width = 50) {
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(width * progress);
    std::cout << "\r[";
    for (int i = 0; i < width; ++i)
        std::cout << (i < pos ? "=" : (i == pos ? ">" : " "));
    std::cout << "] " << int(progress * 100.0) << "%";
    std::cout.flush();
}

bool compare_matrix_verbose(const std::vector<std::vector<double>>& a,
                            const std::vector<std::vector<double>>& b,
                            const std::string& name, double tol = 1e-9) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        std::cout << "❌ " << name << " shape mismatch!\n";
        return false;
    }

    bool match = true;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[0].size(); ++j)
            if (std::abs(a[i][j] - b[i][j]) > tol) {
                std::cout << "❌ " << name << " mismatch at [" << i << "][" << j << "]: "
                          << a[i][j] << " vs " << b[i][j] << "\n";
                match = false;
            }
    return match;
}

bool compare_vector_verbose(const std::vector<double>& a,
                            const std::vector<double>& b,
                            const std::string& name, double tol = 1e-9) {
    if (a.size() != b.size()) {
        std::cout << "❌ " << name << " size mismatch!\n";
        return false;
    }

    bool match = true;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - b[i]) > tol) {
            std::cout << "❌ " << name << " mismatch at [" << i << "]: "
                      << a[i] << " vs " << b[i] << "\n";
            match = false;
        }
    return match;
}

int main() {
    std::cout << "📦 Loading MNIST data...\n";

    std::vector<Image> all_images = load_csv_images("../MNIST/train_images.csv");
    std::vector<int> all_labels = load_csv_labels("../MNIST/train_labels.csv");

    std::vector<Image> x_train_flat, x_test_flat;
    std::vector<int> y_train, y_test;

    select_balanced_subset(all_images, all_labels, x_train_flat, y_train, 500);
    select_balanced_subset(all_images, all_labels, x_test_flat, y_test, 10);

    auto to_tensor = [](const std::vector<Image>& data) {
        Tensor4D out(data.size(), std::vector<std::vector<std::vector<double>>>(
                                      1, std::vector<std::vector<double>>(
                                             28, std::vector<double>(28, 0.0))));
        for (size_t i = 0; i < data.size(); ++i)
            for (int r = 0; r < 28; ++r)
                for (int c = 0; c < 28; ++c)
                    out[i][0][r][c] = data[i][r * 28 + c];
        return out;
    };

    Tensor4D x_train = to_tensor(x_train_flat);
    Tensor4D x_test = to_tensor(x_test_flat);

    CNN model;
    double lr = 0.01;
    int epochs = 1;
    int batch_size = 64;

    std::cout << "🚀 Starting training...\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "\n📘 Epoch " << (epoch + 1) << "/" << epochs << "\n";

        std::vector<int> indices(x_train.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

        Tensor4D x_train_shuffled;
        std::vector<int> y_train_shuffled;
        for (int idx : indices) {
            x_train_shuffled.push_back(x_train[idx]);
            y_train_shuffled.push_back(y_train[idx]);
        }

        size_t steps = (x_train.size() + batch_size - 1) / batch_size;
        for (size_t step = 0; step < steps; ++step) {
            size_t i = step * batch_size;
            size_t end = std::min(i + batch_size, x_train.size());
            Tensor4D x_batch(x_train_shuffled.begin() + i, x_train_shuffled.begin() + end);
            std::vector<int> y_batch(y_train_shuffled.begin() + i, y_train_shuffled.begin() + end);

            model.forward(x_batch, y_batch);
            model.backward(lr);
            print_progress_bar(step + 1, steps);
        }
    }

    std::cout << "\n💾 Saving model to 'trained_model'...\n";
    save_model(model, "trained_model");
    return 0;
}
