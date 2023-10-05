#pragma once

namespace kaminpar::tag {
struct Parallel {};
constexpr inline Parallel par{};

struct Sequential {};
constexpr inline Sequential seq{};

struct Mandatory {};
} // namespace kaminpar::tag
