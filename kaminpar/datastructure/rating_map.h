/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "datastructure/fast_reset_array.h"
#include "datastructure/fixed_size_sparse_map.h"
#include "datastructure/sparse_map.h"

namespace kaminpar {
template<typename Value>
class RatingMap {
  using SuperSmallMap = FixedSizeSparseMap<NodeID, Value, 128>;
  using SmallMap = FixedSizeSparseMap<NodeID, Value>;
  using LargeMap = FastResetArray<Value>;
  //using LargeMap = SparseMap<NodeID, Value>;

public:
  enum class MapType { SUPER_SMALL, SMALL, LARGE };

  explicit RatingMap(const std::size_t max_size) : _max_size{max_size} {}

  MapType update_upper_bound_size(const std::size_t upper_bound_size) {
    select_map(upper_bound_size);
    return _selected_map;
  }

  template<typename F1, typename F2>
  decltype(auto) run_with_map(F1 &&f1, F2 &&f2) {
    switch (_selected_map) {
      case MapType::SUPER_SMALL: return f1(_super_small_map);
      case MapType::SMALL: return f1(_small_map);
      case MapType::LARGE: return f2(_large_map);
    }
    __builtin_unreachable();
  }

  [[nodiscard]] std::size_t small_map_counter() const { return _small_map_counter; }
  [[nodiscard]] std::size_t super_small_map_counter() const { return _super_small_map_counter; }
  [[nodiscard]] std::size_t large_map_counter() const { return _large_map_counter; }

private:
  void select_map(const std::size_t upper_bound_size) {
    if (upper_bound_size < SuperSmallMap::MAP_SIZE / 3) {
      _selected_map = MapType::SUPER_SMALL;
      ++_super_small_map_counter;
    } else if (_max_size < SmallMap::MAP_SIZE || upper_bound_size > SmallMap::MAP_SIZE / 3) {
      _selected_map = MapType::LARGE;
      ++_large_map_counter;
    } else {
      _selected_map = MapType::SMALL;
      ++_small_map_counter;
    }

    if (_selected_map == MapType::LARGE && _large_map.capacity() == 0) { _large_map.resize(_max_size); }
  }

  std::size_t _max_size;
  MapType _selected_map{MapType::SMALL};
  SuperSmallMap _super_small_map{};
  SmallMap _small_map{};
  LargeMap _large_map{}; // allocate on demand

  std::size_t _small_map_counter{0};
  std::size_t _super_small_map_counter{0};
  std::size_t _large_map_counter{0};
};
} // namespace kaminpar