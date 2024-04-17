#pragma once

#include <string>

namespace kaminpar::testing {
inline std::string test_instance(const std::string &name) {
  using namespace std::literals;
  return "test_instances/"s + name;
}
} // namespace kaminpar::testing
