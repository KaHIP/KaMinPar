/*******************************************************************************
 * Utility functions for constant expressions.
 *
 * @file:   constexpr_utils.h
 * @author: Daniel Salwasser
 * @date:   29.12.2023
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <utility>

// Utility functions for constexpr loops based on https://stackoverflow.com/a/47563100
template <std::size_t N> struct Number {
  static const constexpr auto value = N;
};

template <class Lambda, std::size_t... Is>
void constexpr constexpr_for(Lambda &&l, std::index_sequence<Is...>) {
  (l(Number<Is>::value), ...);
}

template <std::size_t N, typename Lambda> constexpr void constexpr_for(Lambda &&l) {
  constexpr_for(std::forward<Lambda>(l), std::make_index_sequence<N>());
}
