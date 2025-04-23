#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <filesystem>

using Image = std::vector<double>; // Keep using double
using Label = int;

// ─────────────────────────────────────────────────────────────────────────────
// Load all images from CSV
std::vector<Image> load_csv_images(const std::string& filename) {
    std::vector<Image> data;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        Image row;
        while (std::getline(ss, val, ',')) {
            try {
                row.push_back(std::stod(val)); // Use stod to read directly into double
            } catch (...) {
                row.push_back(0.0); // Use 0.0 for double
            }
        }
        data.push_back(row);
    }
    return data;
}

// Load all labels from CSV
std::vector<Label> load_csv_labels(const std::string& filename) {
    std::vector<Label> labels;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        try {
            labels.push_back(std::stoi(line));
        } catch (...) {
            labels.push_back(0);
        }
    }
    return labels;
}

// Balanced subset selection
void select_balanced_subset(const std::vector<Image>& all_images, const std::vector<Label>& all_labels,
                            std::vector<Image>& selected_images, std::vector<Label>& selected_labels,
                            int per_class) {
    std::unordered_map<int, int> count;
    for (size_t i = 0; i < all_images.size(); ++i) {
        int label = all_labels[i];
        if (count[label] < per_class) {
            selected_images.push_back(all_images[i]);
            selected_labels.push_back(label);
            count[label]++;
        }
        if (count.size() == 10) {
            bool done = true;
            for (auto& [digit, cnt] : count) {
                if (cnt < per_class) {
                    done = false;
                    break;
                }
            }
            if (done) break;
        }
    }
}

// Save images to PNG
void save_images(const std::vector<Image>& images, const std::vector<Label>& labels,
                 int num_images = 10, const std::string& output_dir = "./output_images/") {
    std::filesystem::create_directories(output_dir);
    for (int i = 0; i < std::min(num_images, static_cast<int>(images.size())); ++i) {
        // 1. Create a cv::Mat from the image data.
        cv::Mat img(28, 28, CV_8U); // Use CV_8U for grayscale images
        for (int r = 0; r < 28; ++r) {
            for (int c = 0; c < 28; ++c) {
                // Scale the normalized pixel value (0-1) to the range 0-255.
                int pixel_value = static_cast<int>(images[i][r * 28 + c] * 255.0);
                // Ensure pixel values are within the valid range [0, 255]
                pixel_value = std::max(0, std::min(255, pixel_value)); // Clamp to [0, 255] -  important
                img.at<uchar>(r, c) = static_cast<uchar>(pixel_value);
            }
        }

        std::string filename = output_dir + "img_" + std::to_string(i) + "_label_" + std::to_string(labels[i]) + ".png";
        cv::imwrite(filename, img);
        std::cout << "✅ Saved: " << filename << std::endl;
    }
}

#endif
