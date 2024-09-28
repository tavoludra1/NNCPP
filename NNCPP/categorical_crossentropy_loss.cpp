//
//  categorical_crossentropy_loss.cpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
//

#include "categorical_crossentropy_loss.hpp"
#include <cmath> // for log
#include <algorithm> // for max, min
#include <stdexcept> // for runtime_error

std::vector<double> Loss_CategoricalCrossentropy::forward(
    const std::vector<std::vector<double>>& y_pred,
    const std::vector<uint8_t>& y_true
) {
    if (y_pred.size() != y_true.size()) {
        throw std::runtime_error("y_pred and y_true must have the same number of samples.");
    }

    size_t samples = y_pred.size();
    std::vector<double> negative_log_likelihoods(samples);

    for (size_t i = 0; i < samples; ++i) {
        // Clip predictions to avoid division by 0
        std::vector<double> y_pred_clipped(y_pred[i].size());
        for (size_t j = 0; j < y_pred[i].size(); ++j) {
            y_pred_clipped[j] = std::max(1e-7, std::min(y_pred[i][j], 1 - 1e-7));
        }

        double correct_confidence;
        if (y_true.size() == samples) { // Categorical labels
            correct_confidence = y_pred_clipped[y_true[i]];
        } else if (y_true.size() == samples * y_pred[0].size()) { // One-hot encoded labels
            correct_confidence = 0.0;
            for (size_t j = 0; j < y_pred[0].size(); ++j) {
                correct_confidence += y_pred_clipped[j] * (y_true[i * y_pred[0].size() + j] ? 1.0 : 0.0);
            }
        } else {
            throw std::runtime_error("Invalid shape for y_true.");
        }

        negative_log_likelihoods[i] = -std::log(correct_confidence);
    }

    return negative_log_likelihoods;
}

double Loss_CategoricalCrossentropy::calculate(
    const std::vector<std::vector<double>>& y_pred,
    const std::vector<uint8_t>& y_true
) {
    std::vector<double> sample_losses = forward(y_pred, y_true);

    double data_loss = 0.0;
    for (double loss : sample_losses) {
        data_loss += loss;
    }
    data_loss /= sample_losses.size();

    return data_loss;
}
