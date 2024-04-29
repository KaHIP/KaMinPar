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

namespace kaminpar {

/*!
 * Invokes a function either directly or indirectly.
 *
 * @tparam direct Whether to call the function directly.
 * @tparam Lambda The type of the lambda to pass to the function.
 * @tparam Function The type of the function to invoke.
 * @param l The lambda to pass to the function.
 * @param fun The function to invoke.
 */
template <bool direct, typename Lambda, typename Function>
constexpr void invoke_indirect(Lambda &&l, Function &&fun) {
  if constexpr (direct) {
    return fun(std::forward<Lambda>(l));
  } else {
    l([&](auto &&l2) { fun(std::forward<decltype(l2)>(l2)); });
  }
}

/*!
 * Invokes a function either directly or indirectly and returns its return value.
 *
 * @tparam direct Whether to call the function directly.
 * @tparam Value The type of the return value of the function.
 * @tparam Lambda The type of the lambda to pass to the function.
 * @tparam Function The type of the function to invoke.
 * @param l The lambda to pass to the function.
 * @param fun The function to invoke.
 */
template <bool direct, typename Value, typename Lambda, typename Function>
constexpr Value invoke_indirect2(Lambda &&l, Function &&fun) {
  if constexpr (direct) {
    return fun(std::forward<Lambda>(l));
  } else {
    Value val;
    l([&](auto &&l2) { val = fun(std::forward<decltype(l2)>(l2)); });
    return val;
  }
}

// Utility functions for constexpr loops based on https://stackoverflow.com/a/47563100
template <std::size_t N> struct Number {
  static const constexpr auto value = N;
};

template <class Lambda, std::size_t... Is>
constexpr void constexpr_for(Lambda &&l, std::index_sequence<Is...>) {
  (l(Number<Is>::value), ...);
}

/*!
 * Calls a lambda a specific amount of times with an index.
 *
 * @tparam N The amount of times to call a lambda.
 * @tparam Lambda The type of lambda to call.
 * @param l The lambda to call N times with the current number of times called.
 */
template <std::size_t N, typename Lambda> constexpr void constexpr_for(Lambda &&l) {
  constexpr_for(std::forward<Lambda>(l), std::make_index_sequence<N>());
}

} // namespace kaminpar
