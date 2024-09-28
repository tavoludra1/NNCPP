//
//  activation_relu.hpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
//

#ifndef ACTIVATION_RELU_H
#define ACTIVATION_RELU_H

#include <vector>
#include <stdio.h>

class Activation_ReLU {
public:
    void forward(const std::vector<std::vector<double>>& inputs);

    std::vector<std::vector<double>> output;
};

#endif // ACTIVATION_RELU_H
