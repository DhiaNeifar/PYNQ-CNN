#ifndef TRAIN_H
#define TRAIN_H

#include "model.h"
#include <iostream>
#include <numeric>  // for std::iota
#include <cmath>

// Compute accuracy
double compute_accuracy(const std::vector<int>& preds, const std::vector<int>& labels) {
    int correct = 0;
    for (size_t i = 0; i < preds.size(); ++i)
        if (preds[i] == labels[i])
            ++correct;
    return static_cast<double>(correct) / preds.size();
}

// Training function
void train(CNN& model,
           const Tensor4D& x_train, const std::vector<int>& y_train,
           const Tensor4D& x_test,  const std::vector<int>& y_test,
           int epochs = 5, double lr = 0.01) {

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double loss = model.forward(x_train, y_train);
        std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss << std::endl;

        model.backward(lr);

        std::vector<int> preds = model.predict(x_test);
        double acc = compute_accuracy(preds, y_test);
        std::cout << "Test Accuracy: " << acc * 100.0 << "%" << std::endl;
    }
}

#endif
