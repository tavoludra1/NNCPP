//
//  layer_dense.hpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
//

#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H

#include <vector>
#include <stdio.h>

class Layer_Dense {
public:
    Layer_Dense(size_t n_inputs, size_t n_neurons);
    void forward(const std::vector<std::vector<double>>& inputs);

    std::vector<std::vector<double>> output;

private:
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> biases;
};

#endif // LAYER_DENSE_H
