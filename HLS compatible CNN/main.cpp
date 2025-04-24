//
// Created by dhianeifar on 23/04/25.
//

#include "model.hpp"
#include "train.hpp"

int main() {
    // Dummy input: 1 image, 1 channel, 28x28
    Tensor4D x_train(1, std::vector<std::vector<std::vector<double>>>(
        1, std::vector<std::vector<double>>(28, std::vector<double>(28, 0.5))));

    std::vector<int> y_train = {1};  // Target label

    // Use the same for testing
    Tensor4D x_test = x_train;
    std::vector<int> y_test = y_train;

    CNN model;

    // Run training (you can change epochs/lr)
    train(model, x_train, y_train, x_test, y_test, 1, 0.01);

    return 0;
}
