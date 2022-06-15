/*******************************************************************************
 * @file:   math.h
 *
 * @author: Daniel Seemaier
 * @date:   14.06.2022
 * @brief:  Utility functions for common (but simple) computations.
 ******************************************************************************/
#pragma once

#include <cmath>
#include <utility>
#include <math.h>

namespace kaminpar {
template <typename Int>
Int encode_virtual_square_position(const Int row, const Int column, const Int num_elements) {
    const Int floor_sqrt              = static_cast<Int>(std::floor(std::sqrt(num_elements)));
    const Int num_additional_elements = num_elements - floor_sqrt * floor_sqrt;
    const Int num_long_rows           = std::min(num_additional_elements, row);
    const Int num_short_rows          = row - num_long_rows;
    return column + (floor_sqrt + 1) * num_long_rows + floor_sqrt * num_short_rows;
}

template <typename Int>
std::pair<Int, Int> decode_virtual_square_position(const Int position, const Int num_elements) {
    const Int floor_sqrt              = static_cast<Int>(std::floor(std::sqrt(num_elements)));
    const Int num_additional_elements = num_elements - floor_sqrt * floor_sqrt;
    const Int num_long_rows           = std::min(floor_sqrt, num_additional_elements);
    if (position <= num_long_rows * (floor_sqrt + 1)) {
        return {position / (floor_sqrt + 1), position % (floor_sqrt + 1)};
    } else {
        const Int position_2 = position - num_long_rows * (floor_sqrt + 1);
        return {num_long_rows + position_2 / floor_sqrt, position_2 % floor_sqrt};
    }
}
} // namespace kaminpar
