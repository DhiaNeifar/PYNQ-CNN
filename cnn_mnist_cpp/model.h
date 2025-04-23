#ifndef MODEL_H
#define MODEL_H

#include "layers.h" // Include the layers.h file
#include "loss.h"   // Include the loss.h file
#include <vector>
#include <algorithm>
#include <iostream>

class CNN {
public:
    Conv2D c1;
    ReLU r1;
    MaxPool2D p1;
    Flatten flat;
    Dense fc2;
    SoftmaxCrossEntropy loss_fn;

    std::vector<std::vector<double>> logits;

    CNN()
        : c1(1, 10, 3),       // 1 input channel, 10 filters, 3x3 kernel
          r1(),
          p1(),
          fc2(13 * 13 * 10, 10) // ⬅️ After Conv(26x26) → MaxPool(13x13), 10 channels
    {}

    double forward(const Tensor4D& x, const std::vector<int>& y) {
        Tensor4D out = c1.forward(x);
        out = r1.forward(out);
        out = p1.forward(out);
        std::vector<std::vector<double>> flat_out = flat.forward(out);
        logits = fc2.forward(flat_out);
        double loss = loss_fn.forward(logits, y);
        return loss;
    }

    void backward(double lr) {
        auto grad = loss_fn.backward();
        grad = fc2.backward(grad, lr);
        Tensor4D grad4D = flat.backward(grad);
        grad4D = p1.backward(grad4D, lr);
        grad4D = r1.backward(grad4D, lr);
        c1.backward(grad4D, lr);
    }

    std::vector<int> predict(const Tensor4D& x) {
        Tensor4D out = c1.forward(x);
        out = r1.forward(out);
        out = p1.forward(out);
        auto flat_out = flat.forward(out);
        auto out3 = fc2.forward(flat_out);

        std::vector<int> predictions(out3.size());
        for (size_t i = 0; i < out3.size(); ++i) {
            auto& row = out3[i];
            predictions[i] = std::distance(row.begin(), std::max_element(row.begin(), row.end()));
        }
        return predictions;
    }
};

#endif
