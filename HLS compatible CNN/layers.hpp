#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <hls_math.h>
#include <ap_fixed.h>
#include <ap_int.h>

#define BATCH 1
#define IN_CHANNELS 1
#define OUT_CHANNELS 10
#define IN_HEIGHT 28
#define IN_WIDTH 28
#define KERNEL_SIZE 3
#define OUT_HEIGHT (IN_HEIGHT - KERNEL_SIZE + 1)
#define OUT_WIDTH (IN_WIDTH - KERNEL_SIZE + 1)
#define POOL_SIZE 2
#define FLAT_SIZE (OUT_CHANNELS * (OUT_HEIGHT / POOL_SIZE) * (OUT_WIDTH / POOL_SIZE))
#define HIDDEN_SIZE 128
#define NUM_CLASSES 10

// ─────────────────────────────────────────────
typedef ap_fixed<16,6> data_t;
typedef data_t Tensor4D[BATCH][IN_CHANNELS][IN_HEIGHT][IN_WIDTH];
typedef data_t Tensor4D_Out[BATCH][OUT_CHANNELS][OUT_HEIGHT][OUT_WIDTH];
typedef data_t Pooled4D[BATCH][OUT_CHANNELS][OUT_HEIGHT/POOL_SIZE][OUT_WIDTH/POOL_SIZE];
typedef data_t Matrix1D[BATCH][FLAT_SIZE];
typedef data_t Matrix2D[BATCH][HIDDEN_SIZE];
typedef data_t Output2D[BATCH][NUM_CLASSES];

// ─────────────────────────────────────────────
void conv2d(Tensor4D &input,
            Tensor4D_Out &output,
            data_t weights[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
            data_t biases[OUT_CHANNELS]) {
#pragma HLS INLINE off
    for (int b = 0; b < BATCH; ++b)
        for (int oc = 0; oc < OUT_CHANNELS; ++oc)
            for (int i = 0; i < OUT_HEIGHT; ++i)
                for (int j = 0; j < OUT_WIDTH; ++j) {
#pragma HLS PIPELINE II=1
                    data_t sum = biases[oc];
                    for (int ic = 0; ic < IN_CHANNELS; ++ic)
                        for (int ki = 0; ki < KERNEL_SIZE; ++ki)
                            for (int kj = 0; kj < KERNEL_SIZE; ++kj)
                                sum += input[b][ic][i + ki][j + kj] * weights[oc][ic][ki][kj];
                    output[b][oc][i][j] = sum;
                }
}

// ─────────────────────────────────────────────
void relu4d(Tensor4D_Out &x) {
#pragma HLS INLINE off
    for (int b = 0; b < BATCH; ++b)
        for (int c = 0; c < OUT_CHANNELS; ++c)
            for (int i = 0; i < OUT_HEIGHT; ++i)
                for (int j = 0; j < OUT_WIDTH; ++j) {
#pragma HLS PIPELINE II=1
                    if (x[b][c][i][j] < 0)
                        x[b][c][i][j] = 0;
                }
}

void maxpool2d(Tensor4D_Out &input, Pooled4D &output) {
#pragma HLS INLINE off
    for (int b = 0; b < BATCH; ++b)
        for (int c = 0; c < OUT_CHANNELS; ++c)
            for (int i = 0; i < OUT_HEIGHT / POOL_SIZE; ++i)
                for (int j = 0; j < OUT_WIDTH / POOL_SIZE; ++j) {
#pragma HLS PIPELINE II=1
                    data_t max_val = input[b][c][i * POOL_SIZE][j * POOL_SIZE];
                    for (int pi = 0; pi < POOL_SIZE; ++pi)
                        for (int pj = 0; pj < POOL_SIZE; ++pj) {
                            data_t val = input[b][c][i * POOL_SIZE + pi][j * POOL_SIZE + pj];
                            if (val > max_val) max_val = val;
                        }
                    output[b][c][i][j] = max_val;
                }
}

void flatten(Pooled4D &input, Matrix1D &output) {
#pragma HLS INLINE off
    for (int b = 0; b < BATCH; ++b) {
        int idx = 0;
        for (int c = 0; c < OUT_CHANNELS; ++c)
            for (int i = 0; i < OUT_HEIGHT / POOL_SIZE; ++i)
                for (int j = 0; j < OUT_WIDTH / POOL_SIZE; ++j)
                    output[b][idx++] = input[b][c][i][j];
    }
}

void dense(Matrix1D &input, Matrix2D &output, data_t weights[FLAT_SIZE][HIDDEN_SIZE], data_t biases[HIDDEN_SIZE]) {
#pragma HLS INLINE off
    for (int b = 0; b < BATCH; ++b)
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
#pragma HLS PIPELINE II=1
            data_t sum = biases[j];
            for (int i = 0; i < FLAT_SIZE; ++i)
                sum += input[b][i] * weights[i][j];
            output[b][j] = sum;
        }
}

void relu2d(Matrix2D &x) {
#pragma HLS INLINE off
    for (int b = 0; b < BATCH; ++b)
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
#pragma HLS PIPELINE II=1
            if (x[b][i] < 0) x[b][i] = 0;
        }
}

void dense_output(Matrix2D &input, Output2D &output, data_t weights[HIDDEN_SIZE][NUM_CLASSES], data_t biases[NUM_CLASSES]) {
#pragma HLS INLINE off
    for (int b = 0; b < BATCH; ++b)
        for (int j = 0; j < NUM_CLASSES; ++j) {
#pragma HLS PIPELINE II=1
            data_t sum = biases[j];
            for (int i = 0; i < HIDDEN_SIZE; ++i)
                sum += input[b][i] * weights[i][j];
            output[b][j] = sum;
        }
}

#endif // LAYERS_HPP
