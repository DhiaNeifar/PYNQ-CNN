#ifndef UTILS_H
#define UTILS_H

#include "model.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <iomanip>

// ─────────────────────────────────────────────
// Save a 1D vector
void save_vector(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot open file: " + filename);
    out << vec.size() << "\n";
    out << std::setprecision(17);
    for (double val : vec) out << val << " ";
    out << "\n";
    out.close();
}

// Load a 1D vector
std::vector<double> load_vector(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open file: " + filename);
    int size;
    in >> size;
    std::vector<double> vec(size);
    for (int i = 0; i < size; ++i) in >> vec[i];
    return vec;
}

// ─────────────────────────────────────────────
// Save a 2D matrix
void save_matrix(const std::vector<std::vector<double>>& mat, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot open file: " + filename);
    out << mat.size() << " " << (mat.empty() ? 0 : mat[0].size()) << "\n";
    out << std::setprecision(17);
    for (const auto& row : mat) {
        for (double val : row) out << val << " ";
        out << "\n";
    }
    out.close();
}

// Load a 2D matrix
std::vector<std::vector<double>> load_matrix(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open file: " + filename);
    int rows, cols;
    in >> rows >> cols;
    std::vector<std::vector<double>> mat(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            in >> mat[i][j];
    return mat;
}

// ─────────────────────────────────────────────
// Save a 4D tensor (for Conv2D weights)
void save_tensor4d(const Tensor4D& tensor, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot open file: " + filename);
    int d1 = tensor.size();
    int d2 = tensor[0].size();
    int d3 = tensor[0][0].size();
    int d4 = tensor[0][0][0].size();
    out << d1 << " " << d2 << " " << d3 << " " << d4 << "\n";
    out << std::setprecision(17);
    for (const auto& a : tensor)
        for (const auto& b : a)
            for (const auto& c : b)
                for (double val : c)
                    out << val << " ";
    out.close();
}

// Load a 4D tensor (for Conv2D weights)
Tensor4D load_tensor4d(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open file: " + filename);
    int d1, d2, d3, d4;
    in >> d1 >> d2 >> d3 >> d4;
    Tensor4D tensor(d1,
        std::vector<std::vector<std::vector<double>>>(
            d2, std::vector<std::vector<double>>(
                d3, std::vector<double>(d4)
            )
        )
    );
    for (int i = 0; i < d1; ++i)
        for (int j = 0; j < d2; ++j)
            for (int k = 0; k < d3; ++k)
                for (int l = 0; l < d4; ++l)
                    in >> tensor[i][j][k][l];
    return tensor;
}

// ─────────────────────────────────────────────
// Save full model (Conv2D + Dense)
void save_model(const CNN& model, const std::string& prefix) {

    save_tensor4d(model.c1.weights, prefix + "_c1_weights.txt");
    save_vector(model.c1.biases,    prefix + "_c1_biases.txt");

    save_matrix(model.fc1.weights, prefix + "_fc1_weights.txt");
    save_vector(model.fc1.biases,  prefix + "_fc1_biases.txt");

    save_matrix(model.fc2.weights, prefix + "_fc2_weights.txt");
    save_vector(model.fc2.biases,  prefix + "_fc2_biases.txt");
}

// Load full model (Conv2D + Dense)
void load_model(CNN& model, const std::string& prefix) {
    model.c1.weights = load_tensor4d(prefix + "_c1_weights.txt");
    model.c1.biases  = load_vector(prefix + "_c1_biases.txt");

    model.fc1.weights = load_matrix(prefix + "_fc1_weights.txt");
    model.fc1.biases  = load_vector(prefix + "_fc1_biases.txt");

    model.fc2.weights = load_matrix(prefix + "_fc2_weights.txt");
    model.fc2.biases  = load_vector(prefix + "_fc2_biases.txt");
}

#endif // UTILS_H
