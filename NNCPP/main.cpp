//
//  main.cpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
// int main(int argc, const char * argv[])

#include <iostream>
#include "spiral_data.hpp"
#include "layer_dense.hpp"
#include "activation_relu.hpp"
#include "activation_softmax.hpp"
#include "categorical_crossentropy_loss.hpp"


int main() {
    // Create dataset
    size_t samples = 100;
    size_t classes = 3;
    auto [X, y] = spiral_data(samples, classes);

    // Create Dense layer
    Layer_Dense dense1(2, 3);

    // Create ReLU and Softmax activations
    Activation_ReLU activation1;
    
    // Create Dense layer 2
    Layer_Dense dense2(3, 3);
    
    Activation_Softmax activation2;
    
    // 3. Create loss object
    Loss_CategoricalCrossentropy loss;

    // Perform a forward pass through the layers and activations
    dense1.forward(X);
    activation1.forward(dense1.output);
    
    // Assuming you have a second dense layer 'dense2'
    dense2.forward(activation1.output);
    activation2.forward(dense2.output);
    
    double current_loss = loss.calculate(activation2.output, y);
    

    // Print output of the first few samples after softmax
    std::cout << "Output of the first 5 samples after Softmax activation:\n";
    for (size_t i = 0; i < 5; ++i) {
        for (double val : activation2.output[i]) {
                std::cout << val << " ";
            }
        std::cout << std::endl;
    }
    
    std::cout << "Loss: " << current_loss << std::endl;
    return 0;
}
