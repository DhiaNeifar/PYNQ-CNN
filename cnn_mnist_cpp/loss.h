#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm> // Needed for std::max_element

class SoftmaxCrossEntropy {
public:
    std::vector<std::vector<double>> probs; // Use double
    std::vector<int> y;

    double forward(const std::vector<std::vector<double>>& logits, const std::vector<int>& labels) { // Use double
        y = labels;
        int batch_size = logits.size();
        int num_classes = logits[0].size();
        probs.resize(batch_size, std::vector<double>(num_classes)); // Use double

        double loss = 0.0; // Use double

        for (int i = 0; i < batch_size; ++i) {
            double max_logit = *std::max_element(logits[i].begin(), logits[i].end()); // Use double

            double sum_exp = 0.0; // Use double
            for (int j = 0; j < num_classes; ++j) {
                probs[i][j] = std::exp(logits[i][j] - max_logit);
                sum_exp += probs[i][j];
            }

            for (int j = 0; j < num_classes; ++j)
                probs[i][j] /= sum_exp;

            loss += -std::log(probs[i][labels[i]] + 1e-9); // Use double
        }

        return loss / batch_size;
    }

    std::vector<std::vector<double>> backward() { // Use double
        int batch_size = probs.size();
        int num_classes = probs[0].size();
        std::vector<std::vector<double>> grad = probs; // Use double

        for (int i = 0; i < batch_size; ++i)
            grad[i][y[i]] -= 1.0; // Changed from 1.0f to 1.0

        for (int i = 0; i < batch_size; ++i)
            for (int j = 0; j < num_classes; ++j)
                grad[i][j] /= batch_size;

        return grad;
    }
};

#endif
