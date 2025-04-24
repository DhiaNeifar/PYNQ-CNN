#ifndef MODEL_HPP
#define MODEL_HPP

#include "layers.hpp"
#include "loss.hpp"

class CNN {
public:
    Conv2D c1;
    ReLU r1;
    MaxPool2D p1;
    Flatten flat;

    Dense fc1;
    ReLU2D r2;
    Dense2 fc2;

    SoftmaxCrossEntropy loss_fn;
    Matrix2D logits;

    CNN() :
        c1(1, 10, 3),
        fc1(13 * 13 * 10, 128),
        fc2(128, 10) {}

    data_t forward(const Tensor4D& x, const int y[MAX_BATCH]) {
        Tensor4D out;
        c1.forward(x, out);

        Tensor4D relu_out;
        r1.forward(out, relu_out);

        Tensor4D pool_out;
        p1.forward(relu_out, pool_out);

        Matrix2D flat_out;
        flat.forward(pool_out, flat_out);

        Matrix2D hidden;
        fc1.forward(flat_out, hidden);

        Matrix2D activated;
        r2.forward(hidden, activated);

        fc2.forward(activated, logits);

        return loss_fn.forward(logits, y);
    }

    void backward(data_t lr) {
        Matrix2D grad;
        loss_fn.backward(logits, grad);

        Matrix2D grad_fc2;
        fc2.backward(grad, grad_fc2, lr);

        Matrix2D grad_relu;
        r2.backward(grad_fc2, grad_relu);

        Matrix2D grad_fc1;
        fc1.backward(grad_relu, grad_fc1, lr);

        Tensor4D grad_flat;
        flat.backward(grad_fc1, grad_flat);

        Tensor4D grad_pool;
        p1.backward(grad_flat, grad_pool);

        Tensor4D grad_relu1;
        r1.backward(grad_pool, grad_relu1);

        c1.backward(grad_relu1, lr);
    }

    void predict(const Tensor4D& x, int predictions[MAX_BATCH]) {
        Tensor4D out;
        c1.forward(x, out);

        Tensor4D relu_out;
        r1.forward(out, relu_out);

        Tensor4D pool_out;
        p1.forward(relu_out, pool_out);

        Matrix2D flat_out;
        flat.forward(pool_out, flat_out);

        Matrix2D hidden;
        fc1.forward(flat_out, hidden);

        Matrix2D activated;
        r2.forward(hidden, activated);

        Matrix2D output;
        fc2.forward(activated, output);

        for (int i = 0; i < MAX_BATCH; ++i) {
            int max_index = 0;
            data_t max_val = output[i][0];
            for (int j = 1; j < 10; ++j) {
                if (output[i][j] > max_val) {
                    max_val = output[i][j];
                    max_index = j;
                }
            }
            predictions[i] = max_index;
        }
    }
};

#endif // MODEL_HPP
