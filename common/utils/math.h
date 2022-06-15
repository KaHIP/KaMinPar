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
    const Int floor_sqrt = static_cast<Int>(std::floor(std::sqrt(num_elements)));
    if (row < floor_sqrt) {
        return row * floor_sqrt + column;
    } else {
        return row + floor_sqrt * column;
    }
}

template <typename Int>
std::pair<Int, Int> decode_virtual_square_position(const Int position, const Int num_elements) {
    const Int floor_sqrt = static_cast<Int>(std::floor(std::sqrt(num_elements)));
    if (position / floor_sqrt < floor_sqrt * floor_sqrt) {
        return {position / floor_sqrt, position % floor_sqrt};
    } else {
        return {position % floor_sqrt, position / floor_sqrt};
    }
}
} // namespace kaminpar
