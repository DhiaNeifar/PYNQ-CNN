#ifndef MODEL_H
#define MODEL_H

#include "layers.h"
#include "loss.h"
#include <vector>
#include <algorithm>
#include <iostream>

class CNN {
public:
    Conv2D c1;
    ReLU r1;
    MaxPool2D p1;
    Flatten flat;

    Dense fc1;       // ⬅️ Hidden layer (1690 → 128)
    ReLU2D r2;       // ⬅️ ReLU for hidden layer
    Dense fc2;       // ⬅️ Output layer (128 → 10)

    SoftmaxCrossEntropy loss_fn;
    std::vector<std::vector<double>> logits;

    CNN()
        : c1(1, 10, 3),
          r1(),
          p1(),
          flat(),
          fc1(13 * 13 * 10, 128), // 1690 → 128
          r2(),
          fc2(128, 10)            // 128 → 10
    {}

    double forward(const Tensor4D& x, const std::vector<int>& y) {
        Tensor4D out = c1.forward(x);
        out = r1.forward(out);
        out = p1.forward(out);
        std::vector<std::vector<double>> flat_out = flat.forward(out);
        auto hidden = fc1.forward(flat_out);
        auto activated = r2.forward(hidden);
        logits = fc2.forward(activated);
        return loss_fn.forward(logits, y);
    }

    void backward(double lr) {
        auto grad = loss_fn.backward();
        grad = fc2.backward(grad, lr);
        grad = r2.backward(grad, lr);
        grad = fc1.backward(grad, lr);
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
        auto hidden = fc1.forward(flat_out);
        auto activated = r2.forward(hidden);
        auto out3 = fc2.forward(activated);

        std::vector<int> predictions(out3.size());
        for (size_t i = 0; i < out3.size(); ++i) {
            auto& row = out3[i];
            predictions[i] = std::distance(row.begin(), std::max_element(row.begin(), row.end()));
        }
        return predictions;
    }
};

#endif
