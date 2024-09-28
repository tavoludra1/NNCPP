//
//  spiral_data.cpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
//

#include "spiral_data.hpp"
#include <cmath>
#include <random>

// Helper function to generate linearly spaced values (similar to np.linspace)
std::vector<double> linspace(double start, double stop, size_t num) {
    std::vector<double> result(num);
    double step = (stop - start) / (num - 1);
    for (size_t i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

// Function to create spiral data
std::pair<std::vector<std::vector<double>>, std::vector<uint8_t>> spiral_data(size_t samples, size_t classes) {
    std::vector<std::vector<double>> X(samples * classes, std::vector<double>(2, 0.0));
    std::vector<uint8_t> y(samples * classes, 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 0.2);

    for (size_t class_number = 0; class_number < classes; ++class_number) {
        for (size_t i = 0; i < samples; ++i) {
            size_t idx = samples * class_number + i;

            std::vector<double> r = linspace(0.0, 1.0, samples);
            std::vector<double> t = linspace(class_number * 4, (class_number + 1) * 4, samples);
            for (size_t j = 0; j < samples; ++j) {
                t[j] += dist(gen);
            }

            X[idx][0] = r[i] * std::sin(t[i] * 2.5);
            X[idx][1] = r[i] * std::cos(t[i] * 2.5);
            y[idx] = static_cast<uint8_t>(class_number);
        }
    }

    return std::make_pair(X, y);
}
