#ifndef TRAIN_HPP
#define TRAIN_HPP

#include "model.hpp"
#include <cmath>

// Compute classification accuracy
double compute_accuracy(const int* preds, const int* labels, int size) {
    int correct = 0;
    for (int i = 0; i < size; ++i)
        if (preds[i] == labels[i])
            ++correct;
    return static_cast<double>(correct) / size;
}

// Train the CNN model
void train(CNN& model,
           Tensor4D& x_train, int* y_train,
           Tensor4D& x_test, int* y_test,
           int train_size,
           int test_size,
           int epochs = 5,
           double lr = 0.01) {

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double loss = model.forward(x_train, y_train, train_size);
        model.backward(lr);

        int* preds = model.predict(x_test, test_size);

        double acc = compute_accuracy(preds, y_test, test_size);

        // For HLS, printf is allowed in simulation, not synthesis
        printf("Epoch %d | Loss: %.6f | Test Accuracy: %.2f%%\n",
               epoch + 1, loss, acc * 100.0);
    }
}

#endif // TRAIN_HPP
