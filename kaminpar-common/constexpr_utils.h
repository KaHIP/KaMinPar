/*******************************************************************************
 * Utility functions for constant expressions.
 *
 * @file:   constexpr_utils.h
 * @author: Daniel Salwasser
 * @date:   29.12.2023
 ******************************************************************************/
#pragma once

#include <array>
#include <string_view>
#include <utility>

namespace kaminpar {

/*!
 * Invokes a function either directly or indirectly.
 *
 * @tparam kDirect Whether to call the function directly.
 * @tparam Lambda The type of the lambda to pass to the function.
 * @tparam Function The type of the function to invoke.
 * @param l The lambda to pass to the function.
 * @param fun The function to invoke.
 */
template <bool kDirect, typename Lambda, typename Function>
constexpr void invoke_indirect(Lambda &&l, Function &&fun) {
  if constexpr (kDirect) {
    return fun(std::forward<Lambda>(l));
  } else {
    l([&](auto &&l2) { fun(std::forward<decltype(l2)>(l2)); });
  }
}

/*!
 * Invokes a function either directly or indirectly and returns its return value.
 *
 * @tparam kDirect Whether to call the function directly.
 * @tparam Value The type of the return value of the function.
 * @tparam Lambda The type of the lambda to pass to the function.
 * @tparam Function The type of the function to invoke.
 * @param l The lambda to pass to the function.
 * @param fun The function to invoke.
 */
template <bool kDirect, typename Value, typename Lambda, typename Function>
constexpr Value invoke_indirect2(Lambda &&l, Function &&fun) {
  if constexpr (kDirect) {
    return fun(std::forward<Lambda>(l));
  } else {
    Value val;
    l([&](auto &&l2) { val = fun(std::forward<decltype(l2)>(l2)); });
    return val;
  }
}

// Utility functions for constexpr loops due to the following source:
// https://stackoverflow.com/a/47563100
namespace {
template <std::size_t N> struct Number {
  static inline constexpr auto value = N;
};

template <typename Lambda, std::size_t... Is>
constexpr void constexpr_for(Lambda &&l, std::index_sequence<Is...>) {
  (l(Number<Is>::value), ...);
}
} // namespace

/*!
 * Calls a lambda a specific number of times.
 *
 * @tparam N The number of times to call the lambda.
 * @tparam Lambda The type of lambda to call.
 * @param l The lambda to call N times.
 */
template <std::size_t N, typename Lambda> constexpr void constexpr_for(Lambda &&l) {
  constexpr_for(std::forward<Lambda>(l), std::make_index_sequence<N>());
}

// Utility functions for getting compile time type names due to the following source:
// https://rodusek.com/posts/2021/03/09/getting-an-unmangled-type-name-at-compile-time/
namespace {
template <std::size_t... Is>
consteval auto substring_as_array(std::string_view str, std::index_sequence<Is...>) {
  return std::array{str[Is]...};
}

template <typename T> consteval auto type_name_array() {
#if defined(__clang__)
  constexpr auto prefix = std::string_view{"[T = "};
  constexpr auto suffix = std::string_view{"]"};
  constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(__GNUC__)
  constexpr auto prefix = std::string_view{"with T = "};
  constexpr auto suffix = std::string_view{"]"};
  constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
#elif defined(_MSC_VER)
  constexpr auto prefix = std::string_view{"type_name_array<"};
  constexpr auto suffix = std::string_view{">(void)"};
  constexpr auto function = std::string_view{__FUNCSIG__};
#else
#error Unsupported compiler
#endif

  constexpr auto start = function.find(prefix) + prefix.size();
  constexpr auto end = function.rfind(suffix);

  static_assert(start < end);

  constexpr auto name = function.substr(start, (end - start));
  constexpr auto wo_ptr_name_size = name.back() == '*' ? name.size() - 1 : name.size();
  constexpr auto wo_ref_name_size = name.back() == '&' ? wo_ptr_name_size - 1 : wo_ptr_name_size;

  return substring_as_array(name, std::make_index_sequence<wo_ref_name_size>{});
}

template <typename T> struct TypeNameHolder {
  static inline constexpr auto value = type_name_array<T>();
};
} // namespace

/**
 * Returns the name of a type at compile time.
 *
 * @tparam T The type whose name to return.
 * @return The name of the type
 */
template <typename T> consteval auto type_name() {
  constexpr auto &value = TypeNameHolder<T>::value;
  return std::string_view{value.data(), value.size()};
}

/*!
 * Checks if a list of tags contains a specific tag.
 *
 * @tparam Tag The tag to check for.
 * @tparam Ts The list of tags to check.
 * @return True if the tag is in the list, false otherwise.
 */
template <typename Tag, typename... Ts>
constexpr bool contains_tag_v = (std::is_same_v<Tag, Ts> || ...);
} // namespace kaminpar
