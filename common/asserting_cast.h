#include <limits>
#include <type_traits>

namespace kaminpar {
template <typename To, typename From>
constexpr bool in_range(const From value) noexcept {
    static_assert(std::is_integral_v<From>);
    static_assert(std::is_integral_v<To>);

    // Check that 0 is included in From and To
    static_assert(std::is_signed_v<From> || std::numeric_limits<From>::min() == 0);
    static_assert(std::is_signed_v<To> || std::numeric_limits<To>::min() == 0);

    // Check that From and To can be converted safely to intmax_t
    if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
        static_assert(std::numeric_limits<From>::digits <= std::numeric_limits<std::intmax_t>::digits);
        static_assert(std::numeric_limits<To>::digits <= std::numeric_limits<std::intmax_t>::digits);
    } else {
        static_assert(std::numeric_limits<From>::digits <= std::numeric_limits<std::uintmax_t>::digits);
        static_assert(std::numeric_limits<To>::digits <= std::numeric_limits<std::uintmax_t>::digits);
    }

    // Check if from is inside To's range
    if constexpr (std::is_unsigned_v<From> && std::is_unsigned_v<To>) {
        return static_cast<std::uintmax_t>(value) <= static_cast<std::uintmax_t>(std::numeric_limits<To>::max());
    } else if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
        return static_cast<std::intmax_t>(value) >= static_cast<std::intmax_t>(std::numeric_limits<To>::min())
               && static_cast<std::intmax_t>(value) <= static_cast<std::intmax_t>(std::numeric_limits<To>::max());
    } else if constexpr (std::is_signed_v<From> && std::is_unsigned_v<To>) {
        return (value < 0) ? false
                           : (static_cast<std::uintmax_t>(value)
                              <= static_cast<std::uintmax_t>(std::numeric_limits<To>::max()));
    } else if constexpr (std::is_unsiged_v<From> && std::is_signed_v<To>) {
        return static_cast<std::uintmax_t>(value) <= static_cast<std::uintmax_t>(std::numeric_limits<To>::max());
    }
}

template <typename To, typename From>
To asserting_cast(const From value) {
    KASSERT(in_range<To, From>(value));
    return static_cast<To>(value);
}
} // namespace kaminpar
