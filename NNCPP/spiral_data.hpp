//
//  spiral_data.hpp
//  NNCPP
//
//  Created by GAPT on 28/09/24.
//

#ifndef SPIRAL_DATA_H
#define SPIRAL_DATA_H

#include <stdio.h>

#include <vector>
#include <cstdint> // For uint8_t

std::pair<std::vector<std::vector<double>>, std::vector<uint8_t>> spiral_data(size_t samples, size_t classes);

// Helper function to generate linearly spaced values
std::vector<double> linspace(double start, double stop, size_t num);

#endif // SPIRAL_DATA_H

