#ifndef UTILS_H
#define UTILS_H

#include "model.h" // Include model.h, which now contains the CNN definition
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ───────────────────────────────────────────────
// Helper to save std::vector<std::vector<double>>
void save_matrix(const std::vector<std::vector<double>>& mat, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot open file for writing: " + filename);

    out << mat.size() << " " << (mat.empty() ? 0 : mat[0].size()) << "\n";
    for (const auto& row : mat)
        for (double val : row)  // Changed float to double
            out << val << " ";
    out << "\n";
    out.close();
}

// Helper to save std::vector<double>
void save_vector(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot open file for writing: " + filename);

    out << vec.size() << "\n";
    for (double val : vec) // Changed float to double
        out << val << " ";
    out << "\n";
    out.close();
}

// Helper to load std::vector<std::vector<double>>
std::vector<std::vector<double>> load_matrix(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open file for reading: " + filename);

    int rows, cols;
    in >> rows >> cols;
    std::vector<std::vector<double>> mat(rows, std::vector<double>(cols)); // Changed float to double
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            in >> mat[i][j];
    in.close();
    return mat;
}

// Helper to load std::vector<double>
std::vector<double> load_vector(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open file for reading: " + filename);

    int size;
    in >> size;
    std::vector<double> vec(size); // Changed float to double
    for (int i = 0; i < size; ++i)
        in >> vec[i];
    in.close();
    return vec;
}

// ───────────────────────────────────────────────
// Save model parameters (Dense only)
void save_model(const CNN& model, const std::string& prefix) {
    save_matrix(model.fc2.weights, prefix + "_fc2_weights.txt");
    save_vector(model.fc2.biases, prefix + "_fc2_biases.txt");
}

// Load model parameters (Dense only)
void load_model(CNN& model, const std::string& prefix) { // Remove the const
    model.fc2.weights = load_matrix(prefix + "_fc2_weights.txt");
    model.fc2.biases  = load_vector(prefix + "_fc2_biases.txt");
}

#endif
