//
//  activation_relu.cpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
//

#include "activation_relu.hpp"
#include <algorithm> // for std::max

void Activation_ReLU::forward(const std::vector<std::vector<double>>& inputs) {
    size_t num_samples = inputs.size();
    size_t num_features = inputs[0].size();
    output.resize(num_samples, std::vector<double>(num_features));

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            output[i][j] = std::max(0.0, inputs[i][j]);
        }
    }
}
