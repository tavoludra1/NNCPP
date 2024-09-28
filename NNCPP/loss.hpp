//
//  loss.hpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
//

#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include <stdio.h>

class Loss {
public:
    virtual double calculate(const std::vector<std::vector<double>>& output,
                             const std::vector<uint8_t>& y) = 0; // Pure virtual function

protected:
    // This will be implemented by derived classes
    virtual std::vector<double> forward(const std::vector<std::vector<double>>& output,
                                        const std::vector<uint8_t>& y) = 0;
};

#endif // LOSS_H
