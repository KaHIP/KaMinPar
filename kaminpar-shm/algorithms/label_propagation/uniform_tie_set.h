/*******************************************************************************
 * Scratch storage for uniform random tie breaking.
 *
 * @file:   uniform_tie_set.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <cstddef>

#include "kaminpar-common/random.h"

namespace kaminpar::shm::lp {

template <typename Container> class UniformTieSet {
public:
  using ID = typename Container::value_type;

  explicit UniformTieSet(Container &ties) : _ties(ties) {}

  void replace_with(const ID id) {
    _ties.clear();
    _ties.push_back(id);
  }

  void add(const ID id) {
    _ties.push_back(id);
  }

  [[nodiscard]] ID select_or(const ID fallback, Random &random) const {
    if (_ties.size() > 1) {
      return _ties[random.random_index(0, _ties.size())];
    }
    return fallback;
  }

  void clear() {
    _ties.clear();
  }

private:
  Container &_ties;
};

} // namespace kaminpar::shm::lp
