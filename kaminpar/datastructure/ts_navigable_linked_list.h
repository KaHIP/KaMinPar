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

#include "kaminpar/parallel.h"

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

namespace kaminpar {
template<typename Key, typename Element>
class LocalNavigableLinkedList {
  using Self = LocalNavigableLinkedList<Key, Element>;
  using MemoryChunk = scalable_vector<Element>;

  static constexpr std::size_t kChunkSize = 1 << 15;

public:
  struct Marker {
    Key key;
    std::size_t position;
    Self *local_list;
  };

  void mark(const Key key) { _markers.push_back({key, position(), this}); }

  void push_back(Element &&e) {
    flush_if_full();
    _current_chunk.push_back(std::move(e));
  }

  void push_back(const Element &e) {
    flush_if_full();
    _current_chunk.push_back(e);
  }

  template<typename... Args>
  void emplace_back(Args &&...args) {
    flush_if_full();
    _current_chunk.emplace_back(std::forward<Args>(args)...);
  }

  [[nodiscard]] std::size_t position() const { return _chunks.size() * kChunkSize + _current_chunk.size(); }

  [[nodiscard]] const Element &get(const std::size_t position) const {
    return _chunks[position / kChunkSize][position % kChunkSize];
  }

  [[nodiscard]] Element &get(const std::size_t position) {
    return const_cast<Element &>(get(static_cast<const decltype(this)>(this)));
  }

  void flush() {
    if (!_current_chunk.empty()) {
      _chunks.push_back(std::move(_current_chunk));
      _current_chunk.clear();
      _current_chunk.reserve(kChunkSize);
    }
  }

  [[nodiscard]] const auto &markers() const { return _markers; }

private:
  void flush_if_full() {
    if (_current_chunk.size() == kChunkSize) { flush(); }
  }

  scalable_vector<MemoryChunk> _chunks;
  MemoryChunk _current_chunk;
  scalable_vector<Marker> _markers;
};

template<typename Key, typename Element>
using NavigableLinkedList = tbb::enumerable_thread_specific<LocalNavigableLinkedList<Key, Element>>;

template<typename Key, typename Element>
using NavigationMarker = typename LocalNavigableLinkedList<Key, Element>::Marker;

namespace ts_navigable_list {
template<typename Key, typename Element>
scalable_vector<NavigationMarker<Key, Element>> combine(NavigableLinkedList<Key, Element> &list,
                                       scalable_vector<NavigationMarker<Key, Element>> global_markers = {}) {
  parallel::IntegralAtomicWrapper<std::size_t> global_pos = 0;
  std::size_t num_markers = 0;
  for (const auto &local_list : list) { num_markers += local_list.markers().size(); }
  if (global_markers.size() < num_markers) { global_markers.resize(num_markers); }

  tbb::parallel_invoke(
      [&] {
        tbb::parallel_for(list.range(), [&](auto &r) {
          for (auto &local_list : r) { local_list.flush(); }
        });
      },
      [&] {
        tbb::parallel_for(list.range(), [&](const auto &r) {
          for (const auto &local_list : r) {
            const auto &markers = local_list.markers();
            const std::size_t local_pos = global_pos.fetch_add(markers.size());
            std::copy(markers.begin(), markers.end(), global_markers.begin() + local_pos);
          }
        });
      });

  return global_markers;
}
} // namespace ts_navigable_list
} // namespace kaminpar