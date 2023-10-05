#pragma once

#include <vector>

#include "kaminpar-shm/context.h"

namespace kaminpar::shm::testing {
template <typename View> auto view_to_vector(const View &&view) {
  std::vector<std::decay_t<decltype(*view.begin())>> vec;
  for (const auto &e : view) {
    vec.push_back(e);
  }
  return vec;
}
} // namespace kaminpar::shm::testing
