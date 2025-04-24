#include "data_loader.h"
#include "model.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "📂 Loading model...\n";
    CNN model;
    load_model(model, "trained_model");

    std::cout << "📦 Loading test data...\n";
    std::vector<Image> test_images = load_csv_images("../MNIST/test_images.csv");
    std::vector<int> test_labels  = load_csv_labels("../MNIST/test_labels.csv");

    // Convert to Tensor4D
    Tensor4D x_test(test_images.size(), std::vector<std::vector<std::vector<double>>>(
        1, std::vector<std::vector<double>>(28, std::vector<double>(28, 0.0))));

    for (size_t i = 0; i < test_images.size(); ++i)
        for (int r = 0; r < 28; ++r)
            for (int c = 0; c < 28; ++c)
                x_test[i][0][r][c] = test_images[i][r * 28 + c];

    std::cout << "🧠 Running inference...\n";
    std::vector<int> predictions = model.predict(x_test);

    // Compute accuracy
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i)
        if (predictions[i] == test_labels[i]) ++correct;

    double accuracy = static_cast<double>(correct) / predictions.size();
    std::cout << "✅ Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100.0 << "%\n";

    // Save predictions + labels to file
    std::ofstream out("evaluation_results.txt");
    out << "Predicted\tTrue\n";
    for (size_t i = 0; i < predictions.size(); ++i)
        out << predictions[i] << "\t" << test_labels[i] << "\n";
    out << "\nAccuracy: " << std::fixed << std::setprecision(4) << accuracy * 100.0 << "%\n";
    out.close();

    std::cout << "📄 Results saved to evaluation_results.txt\n";

    return 0;
}
