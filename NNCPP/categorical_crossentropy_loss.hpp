//
//  categorical_crossentropy_loss.hpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
//

#ifndef CATEGORICAL_CROSSENTROPY_LOSS_H
#define CATEGORICAL_CROSSENTROPY_LOSS_H

#include "loss.hpp"
#include <stdio.h>

class Loss_CategoricalCrossentropy : public Loss {
public:
    double calculate(const std::vector<std::vector<double>>& y_pred,
                     const std::vector<uint8_t>& y_true) override;

protected:
    std::vector<double> forward(const std::vector<std::vector<double>>& y_pred,
                                const std::vector<uint8_t>& y_true) override;
};

#endif // CATEGORICAL_CROSSENTROPY_LOSS_H
