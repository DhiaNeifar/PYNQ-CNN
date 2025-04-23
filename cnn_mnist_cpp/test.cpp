#include "data_loader.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <map> // Use std::map for more efficient counting

int main() {
    std::cout << "📦 Loading MNIST data...\n";

    // 1. Test load_csv_images
    std::cout << "Testing load_csv_images...\n";
    std::vector<Image> all_images = load_csv_images("../MNIST/train_images.csv"); // Adjust the path if needed
    if (all_images.empty()) {
        std::cerr << "Error: load_csv_images returned an empty vector.  Check the file path and data.\n";
        return 1;
    }
    std::cout << "Loaded " << all_images.size() << " images.\n";
    assert(all_images.size() > 0); // Basic sanity check

    // Check the size of the first image.  MNIST images are 28x28 = 784
    if (!all_images.empty() && all_images[0].size() != 784) {
        std::cerr << "Error: First image has incorrect size. Expected 784, got " << all_images[0].size() << "\n";
        return 1;
    }
    std::cout << "First image size check passed.\n";

    // Check some pixel values of the first image.
    std::cout << "Pixel values of the first image (first 10 pixels):\n";
    for (size_t i = 0; i < std::min<size_t>(10, all_images[0].size()); ++i) {
        std::cout << std::fixed << std::setprecision(3) << all_images[0][i] << " ";
    }
    std::cout << std::endl;

    // 2. Test load_csv_labels
    std::cout << "\nTesting load_csv_labels...\n";
    std::vector<Label> all_labels = load_csv_labels("../MNIST/train_labels.csv"); // Adjust the path if needed
    if (all_labels.empty()) {
        std::cerr << "Error: load_csv_labels returned an empty vector. Check the file path and data.\n";
        return 1;
    }
    std::cout << "Loaded " << all_labels.size() << " labels.\n";
    assert(all_labels.size() > 0);

    if (all_images.size() != all_labels.size()) {
        std::cerr << "Error: Number of images and labels do not match.\n";
        return 1;
    }
    std::cout << "Number of images and labels match.\n";

    // Print first 10 labels
    std::cout << "First 10 labels: ";
    for (size_t i = 0; i < std::min<size_t>(10, all_labels.size()); ++i)
        std::cout << all_labels[i] << " ";
    std::cout << std::endl;


    // 3. Test select_balanced_subset
    std::cout << "\nTesting select_balanced_subset...\n";
    std::vector<Image> selected_images;
    std::vector<Label> selected_labels;
    int per_class = 10;
    select_balanced_subset(all_images, all_labels, selected_images, selected_labels, per_class);
    if (selected_images.empty()) {
        std::cerr << "Error: selected image is empty";
        return 1;
    }
    if (selected_labels.empty()) {
        std::cerr << "Error: selected label is empty";
        return 1;
    }
    std::cout << "Selected " << selected_images.size() << " images.\n";
    std::cout << "Selected " << selected_labels.size() << " labels.\n";
    assert(selected_images.size() == 10 * per_class); // 10 classes in MNIST
    assert(selected_labels.size() == 10 * per_class);

    // Check that the subset is balanced
    std::map<int, int> label_counts; // Use std::map
    for (int label : selected_labels) {
        label_counts[label]++;
    }
    bool balanced = true;
    for (int i = 0; i < 10; ++i) {
        if (label_counts[i] != per_class) {
            balanced = false;
            break;
        }
    }
    if (balanced)
        std::cout << "Subset is balanced (" << per_class << " images per class).\n";
    else {
        std::cerr << "Error: Subset is not balanced.\n";
        for (int i = 0; i < 10; ++i)
            std::cout << "Class " << i << ": " << label_counts[i] << " images\n";
        return 1; // Return 1 to indicate failure
    }

    // 4. Test save_images (basic test - visual inspection is best)
    std::cout << "\nTesting save_images...\n";
    save_images(selected_images, selected_labels, 5, "./test_output_images/"); // Save a few images
    std::cout << "Saved 5 images to ./test_output_images/.  Please check the directory to visually verify the output.\n";

    std::cout << "\nData loading tests completed.  Please check for any error messages above and visually inspect the saved images.\n";
    return 0;
}
