//
//  activation_softmax.cpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
//

#include "activation_softmax.hpp"
#include <cmath> // for exp
#include <algorithm> // for max_element

void Activation_Softmax::forward(const std::vector<std::vector<double>>& inputs) {
    size_t num_samples = inputs.size();
    size_t num_features = inputs[0].size();
    output.resize(num_samples, std::vector<double>(num_features));

    for (size_t i = 0; i < num_samples; ++i) {
        // Find the maximum value in the row (similar to np.max(inputs, axis=1))
        double max_val = *std::max_element(inputs[i].begin(), inputs[i].end());

        double sum_exp = 0.0;
        for (size_t j = 0; j < num_features; ++j) {
            // Subtract the max to avoid potential overflow in exp
            output[i][j] = std::exp(inputs[i][j] - max_val);
            sum_exp += output[i][j];
        }

        // Normalize the exponentiated values
        for (size_t j = 0; j < num_features; ++j) {
            output[i][j] /= sum_exp;
        }
    }
}
