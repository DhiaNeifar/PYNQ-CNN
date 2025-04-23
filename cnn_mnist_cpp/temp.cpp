//
// Created by dhianeifar on 23/04/25.
//


#include "data_loader.h"
#include "model.h"
#include "train.h"
#include "utils.h" // Include the updated utils.h
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip> // for std::setw

void print_progress_bar(size_t current, size_t total, int width = 50) {
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(width * progress);

    std::cout << "\r[";
    for (int i = 0; i < width; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%";
    std::cout.flush();
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
        // Change float to double to match Tensor4D definition
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

    CNN model; // Use the namespace ML
    double lr = 0.01;  // Use double for consistency
    int epochs = 5;
    int batch_size = 64;

    std::vector<double> train_loss, train_acc; // Use double

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

        double epoch_loss = 0.0; // Use double
        int correct = 0;
        int total = 0;

        size_t steps = (x_train.size() + batch_size - 1) / batch_size;

        for (size_t step = 0; step < steps; ++step) {
            size_t i = step * batch_size;
            size_t end = std::min(i + batch_size, x_train.size());
            Tensor4D x_batch(x_train_shuffled.begin() + i, x_train_shuffled.begin() + end);
            std::vector<int> y_batch(y_train_shuffled.begin() + i, y_train_shuffled.begin() + end);

            double loss = model.forward(x_batch, y_batch); // Use double
            model.backward(lr);
            // Inspect gradient updates after backward()
            if (step == 0 && epoch == 0) {
                // Print weight before and after update (for fc1)
                static double previous_weight = model.fc1.weights[0][0]; // Use double
                std::cout << "\n🔍 Initial fc1.weights[0][0]: " << previous_weight << std::endl;

                double new_weight = model.fc1.weights[0][0]; // Use double
                std::cout << "🔁 After 1st backward, fc1.weights[0][0]: " << new_weight << std::endl;

                double diff = new_weight - previous_weight; // Use double
                std::cout << "📉 Weight change: " << diff << std::endl;
            }

            // Also inspect softmax outputs
            if (step == 0 && epoch == 0) {
                std::cout << "\n🧠 Softmax output for first sample: ";
                const auto& probs = model.loss_fn.probs[0];
                for (double p : probs) // Use double
                    std::cout << std::fixed << std::setprecision(3) << p << " ";
                std::cout << "\nTarget label: " << y_batch[0] << std::endl;
            }

            epoch_loss += loss;

            std::vector<int> preds = model.predict(x_batch);
            for (size_t j = 0; j < preds.size(); ++j)
                if (preds[j] == y_batch[j])
                    ++correct;
            total += y_batch.size();

            print_progress_bar(step + 1, steps);
        }

        double acc = static_cast<double>(correct) / total; // Use double
        train_loss.push_back(epoch_loss / steps);
        train_acc.push_back(acc);

        std::cout << "\n✅ Epoch " << (epoch + 1) << " finished - Loss: "
                  << std::fixed << std::setprecision(4) << train_loss.back()
                  << ", Accuracy: " << std::fixed << std::setprecision(2)
                  << train_acc.back() * 100.0 << "%\n"; // Use double
    }

    std::cout << "💾 Saving model to 'trained_model'...\n";
    save_model(model, "trained_model"); // Use the namespace ML

    std::cout << "🎉 Done.\n";
    return 0;
}
