#ifndef LOSS_HPP
#define LOSS_HPP

#include <cmath>
#include "ap_int.h"
#include "hls_math.h"

#define BATCH_SIZE 64
#define NUM_CLASSES 10

class SoftmaxCrossEntropy {
public:
    double probs[BATCH_SIZE][NUM_CLASSES];
    int y[BATCH_SIZE];

    double forward(double logits[BATCH_SIZE][NUM_CLASSES], const int labels[BATCH_SIZE]) {
#pragma HLS INLINE off
        double loss = 0.0;

        for (int i = 0; i < BATCH_SIZE; ++i) {
            y[i] = labels[i];
            double max_logit = logits[i][0];
            for (int j = 1; j < NUM_CLASSES; ++j) {
                if (logits[i][j] > max_logit) max_logit = logits[i][j];
            }

            double sum_exp = 0.0;
            for (int j = 0; j < NUM_CLASSES; ++j) {
                probs[i][j] = hls::exp(logits[i][j] - max_logit);
                sum_exp += probs[i][j];
            }

            for (int j = 0; j < NUM_CLASSES; ++j) {
                probs[i][j] /= sum_exp;
            }

            loss += -hls::log(probs[i][labels[i]] + 1e-9);
        }

        return loss / BATCH_SIZE;
    }

    void backward(double grad[BATCH_SIZE][NUM_CLASSES]) {
#pragma HLS INLINE off
        for (int i = 0; i < BATCH_SIZE; ++i) {
            for (int j = 0; j < NUM_CLASSES; ++j) {
                grad[i][j] = probs[i][j];
                if (j == y[i]) grad[i][j] -= 1.0;
                grad[i][j] /= BATCH_SIZE;
            }
        }
    }
};

#endif
