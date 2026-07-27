/*******************************************************************************
 * Thread-local adaptive rating maps used by parallel graph algorithms.
 *
 * @file:   rating_map_pool.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <cstddef>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/algorithms/label_propagation/adaptive_rating_map.h"

#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::shm::lp {

template <
    typename Key,
    typename Value,
    template <typename, typename> typename DirectMap = adaptive_rating_map::FastResetArray>
class RatingMapPool {
public:
  using RatingMap = AdaptiveRatingMap<Key, Value, DirectMap>;

  RatingMapPool() : _rating_maps([] { return RatingMap(1); }) {}
  RatingMapPool(const RatingMapPool &) = delete;
  RatingMapPool &operator=(const RatingMapPool &) = delete;
  RatingMapPool(RatingMapPool &&) = delete;
  RatingMapPool &operator=(RatingMapPool &&) = delete;

  struct Local {
    RatingMap &ratings;
    ScalableVector<Key> &ties;
    ScalableVector<Key> &favored_ties;
  };

  void ensure_capacity(const std::size_t capacity) {
    if (!_rating_maps.empty() && _capacity == capacity) {
      return;
    }

    if (_rating_maps.empty() || _capacity < capacity) {
      _capacity = capacity;
      _rating_maps =
          tbb::enumerable_thread_specific<RatingMap>([this] { return RatingMap(_capacity); });
    } else {
      _capacity = capacity;
      for (RatingMap &map : _rating_maps) {
        map.change_max_size(capacity);
      }
    }
  }

  [[nodiscard]] Local local() {
    return {_rating_maps.local(), _ties.local(), _favored_ties.local()};
  }

  [[nodiscard]] RatingMap &local_ratings() {
    return _rating_maps.local();
  }

  [[nodiscard]] ScalableVector<Key> &local_ties() {
    return _ties.local();
  }

  [[nodiscard]] tbb::enumerable_thread_specific<RatingMap> &maps() {
    return _rating_maps;
  }

  void free() {
    _rating_maps.clear();
    _ties.clear();
    _favored_ties.clear();
    _capacity = 0;
  }

private:
  std::size_t _capacity = 0;
  tbb::enumerable_thread_specific<RatingMap> _rating_maps;
  tbb::enumerable_thread_specific<ScalableVector<Key>> _ties;
  tbb::enumerable_thread_specific<ScalableVector<Key>> _favored_ties;
};

} // namespace kaminpar::shm::lp
