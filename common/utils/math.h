/*******************************************************************************
 * @file:   math.h
 *
 * @author: Daniel Seemaier
 * @date:   17.06.2022
 * @brief:  Small math utility functions
 ******************************************************************************/
#pragma once

#include <cmath>

namespace kaminpar::math {
template <typename Int>
bool is_square(const Int value) {
    const Int sqrt = std::sqrt(value);
    return sqrt * sqrt == value;
}
} // namespace kaminpar::math
