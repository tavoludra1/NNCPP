//
//  activation_softmax.hpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
//

#ifndef ACTIVATION_SOFTMAX_H
#define ACTIVATION_SOFTMAX_H

#include <vector>
#include <cstdint> // For uint8_t
#include <limits> // For numeric_limits
#include <stdio.h>

class Activation_Softmax {
public:
    void forward(const std::vector<std::vector<double>>& inputs);

    std::vector<std::vector<double>> output;
};

#endif // ACTIVATION_SOFTMAX_H
