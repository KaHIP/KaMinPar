/*******************************************************************************
 * Non-addressable growable double-ended priority queue.
 *
 * @file:   dynamic_de_heap.h
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   01.01.2025
 ******************************************************************************/
#pragma once

#include <cstdlib>
#include <vector>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/binary_heap.h"

namespace kaminpar {

template <typename ID, typename Key, template <typename...> typename Container = std::vector>
class DynamicBinaryMinMaxForest {
  struct HeapElement {
    ID complementary_pos;
    ID id;
    Key key;
  };

  template <template <typename> typename Comparator> class DynamicBinaryForest {
    static constexpr std::size_t kTreeArity = 4;
    static constexpr ID kInvalidID = std::numeric_limits<ID>::max();

  public:
    explicit DynamicBinaryForest() {}

    DynamicBinaryForest(const DynamicBinaryForest &) = delete;
    DynamicBinaryForest &operator=(const DynamicBinaryForest &) = delete;

    DynamicBinaryForest(DynamicBinaryForest &&) noexcept = default;
    DynamicBinaryForest &operator=(DynamicBinaryForest &&) noexcept = default;

    Container<HeapElement> *init(const std::size_t num_heaps) {
      _heaps.resize(num_heaps);
      return _heaps.data();
    }

    void init_complementary_data(Container<HeapElement> *complementary_heaps) {
      _complementary_heaps = complementary_heaps;
    }

    [[nodiscard]] bool empty(const std::size_t heap) const {
      return _heaps[heap].empty();
    }

    [[nodiscard]] std::size_t size(const std::size_t heap) const {
      return _heaps[heap].size();
    }

    HeapElement &push(const std::size_t heap, const ID id, const Key key) {
      KASSERT(key != Comparator<Key>::kMinValue);
      KASSERT(_comparator(key, Comparator<Key>::kMinValue));

      const std::size_t initial_pos = size(heap);
      _heaps[heap].emplace_back(0, id, key);

      const std::size_t final_pos = sift_up(heap, initial_pos);
      _heaps[heap][final_pos].complementary_pos = final_pos;

      return _heaps[heap][final_pos];
    }

    [[nodiscard]] const HeapElement &peek(const std::size_t heap) const {
      return _heaps[heap].front();
    }

    [[nodiscard]] ID peek_id(const std::size_t heap) const {
      return _heaps[heap].front().id;
    }

    [[nodiscard]] Key peek_key(const std::size_t heap) const {
      return _heaps[heap].front().key;
    }

    void pop(const std::size_t heap) {
      if (size(heap) == 1) [[unlikely]] {
        _heaps[heap].clear();
        return;
      }

      swap<true>(heap, size(heap) - 1, 0);
      _heaps[heap].pop_back();
      sift_down(heap, 0);
    }

    void remove(const std::size_t heap, const std::size_t pos) {
      _heaps[heap][pos].key = Comparator<Key>::kMinValue;
      sift_up(heap, pos);
      pop(heap);
    }

    void clear() {
      for (std::size_t heap = 0; heap < _heaps.size(); ++heap) {
        _heaps[heap].clear();
      }
    }

  private:
    std::size_t sift_up(const std::size_t heap, std::size_t pos) {
      while (pos != 0) {
        const std::size_t parent_pos = (pos - 1) / kTreeArity;

        if (!_comparator(_heaps[heap][parent_pos].key, _heaps[heap][pos].key)) {
          return pos;
        }

        swap<true>(heap, parent_pos, pos);
        pos = parent_pos;
      }

      return 0;
    }

    void sift_down(const std::size_t heap_id, std::size_t pos) {
      const auto &heap = _heaps[heap_id];
      const std::size_t heap_size = heap.size();

      while (true) {
        const std::size_t first_child = kTreeArity * pos + 1;
        if (first_child >= heap_size) {
          return;
        }

        const std::size_t last_child_p1 = std::min(first_child + kTreeArity, heap_size);

        std::size_t smallest_child = first_child;
        for (std::size_t child = first_child + 1; child < last_child_p1; ++child) {
          if (_comparator(heap[smallest_child].key, heap[child].key)) {
            smallest_child = child;
          }
        }

        if (_comparator(heap[smallest_child].key, heap[pos].key) ||
            heap[smallest_child].key == heap[pos].key) {
          return;
        }

        swap(heap_id, smallest_child, pos);
        pos = smallest_child;
      }
    }

    template <bool kSyncOnlyFirst = false>
    void swap(const std::size_t heap, const std::size_t a, const std::size_t b) {
      if constexpr (kSyncOnlyFirst) {
        _complementary_heaps[heap][_heaps[heap][a].complementary_pos].complementary_pos = b;
      } else {
        std::swap(
            _complementary_heaps[heap][_heaps[heap][a].complementary_pos].complementary_pos,
            _complementary_heaps[heap][_heaps[heap][b].complementary_pos].complementary_pos
        );
      }

      std::swap(_heaps[heap][a], _heaps[heap][b]);
    }

    Container<Container<HeapElement>> _heaps;
    Container<HeapElement> *_complementary_heaps;
    Comparator<Key> _comparator{};
  };

  using DynamicBinaryMaxForest = DynamicBinaryForest<binary_heap::max_heap_comparator>;
  using DynamicBinaryMinForest = DynamicBinaryForest<binary_heap::min_heap_comparator>;

public:
  DynamicBinaryMinMaxForest() {}

  DynamicBinaryMinMaxForest(const DynamicBinaryMinMaxForest &) = delete;
  DynamicBinaryMinMaxForest &operator=(const DynamicBinaryMinMaxForest &) = delete;

  DynamicBinaryMinMaxForest(DynamicBinaryMinMaxForest &&) noexcept = default;
  DynamicBinaryMinMaxForest &operator=(DynamicBinaryMinMaxForest &&) noexcept = default;

  void init(const std::size_t num_heaps) {
    _num_heaps = num_heaps;

    auto *max_forest_data = _max_forest.init(num_heaps);
    auto *min_forest_data = _min_forest.init(num_heaps);

    _max_forest.init_complementary_data(min_forest_data);
    _min_forest.init_complementary_data(max_forest_data);
  }

  [[nodiscard]] std::size_t size(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(_max_forest.size(heap) == _min_forest.size(heap));

    return _max_forest.size(heap);
  }

  [[nodiscard]] bool empty(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(_max_forest.empty(heap) == _min_forest.empty(heap));

    return _max_forest.empty(heap);
  }

  void push(const std::size_t heap, const ID id, const Key key) {
    KASSERT(heap < _num_heaps);

    auto &element_min = _min_forest.push(heap, id, key);
    auto &element_max = _max_forest.push(heap, id, key);
    std::swap(element_min.complementary_pos, element_max.complementary_pos);
  }

  [[nodiscard]] ID peek_min_id(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(!_min_forest.empty(heap));

    return _min_forest.peek_id(heap);
  }

  [[nodiscard]] ID peek_max_id(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(!_max_forest.empty(heap));

    return _max_forest.peek_id(heap);
  }

  [[nodiscard]] Key peek_min_key(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(!_min_forest.empty(heap));

    return _min_forest.peek_key(heap);
  }

  [[nodiscard]] Key peek_max_key(const std::size_t heap) const {
    KASSERT(heap < _num_heaps);
    KASSERT(!_max_forest.empty(heap));

    return _max_forest.peek_key(heap);
  }

  void pop_min(const std::size_t heap) {
    KASSERT(heap < _num_heaps);
    KASSERT(!_min_forest.empty(heap));
    KASSERT(!_max_forest.empty(heap));

    _max_forest.remove(heap, _min_forest.peek(heap).complementary_pos);
    _min_forest.pop(heap);
  }

  void pop_max(const std::size_t heap) {
    KASSERT(heap < _num_heaps);
    KASSERT(!_min_forest.empty(heap));
    KASSERT(!_max_forest.empty(heap));

    _min_forest.remove(heap, _max_forest.peek(heap).complementary_pos);
    _max_forest.pop(heap);
  }

  void clear() {
    _min_forest.clear();
    _max_forest.clear();
  }

private:
  std::size_t _num_heaps;

  DynamicBinaryMinForest _min_forest;
  DynamicBinaryMaxForest _max_forest;
};

} // namespace kaminpar
