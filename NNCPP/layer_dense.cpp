//
//  layer_dense.cpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
//

#include "layer_dense.hpp"
#include <random>

// Constructor for Layer_Dense
Layer_Dense::Layer_Dense(size_t n_inputs, size_t n_neurons)
    : weights(n_neurons, std::vector<double>(n_inputs)),
      biases(1, std::vector<double>(n_neurons, 0.0))
{
    // Initialize weights with random values (scaled by 0.01)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 0.01);

    for (size_t i = 0; i < n_neurons; ++i) {
        for (size_t j = 0; j < n_inputs; ++j) {
            weights[i][j] = dist(gen);
        }
    }
}

// Forward pass implementation
void Layer_Dense::forward(const std::vector<std::vector<double>>& inputs) {
    size_t num_samples = inputs.size();
    output.resize(num_samples, std::vector<double>(biases[0].size(), 0.0)); // Initialize output

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < biases[0].size(); ++j) {
            for (size_t k = 0; k < inputs[0].size(); ++k) {
                output[i][j] += inputs[i][k] * weights[j][k];
            }
            output[i][j] += biases[0][j];
        }
    }
}
