/*******************************************************************************
 * Basic MultiQueue.
 *
 * @file:   multi_queue.h
 * @author: Daniel Seemaier
 * @date:   17.06.2025
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-common/datastructures/dynamic_binary_heap.h"

namespace kaminpar {

template <typename ID, typename Key, template <typename> typename Comparator> class MultiQueue {
  constexpr static std::size_t kNumPQsPerThread = 2;
  constexpr static int kNumPopAttempts = 32;

  struct Token {
    Token(const int seed, const std::size_t num_pqs) : dist(0, num_pqs - 1) {
      rng.seed(seed);
    }

    [[nodiscard]] std::size_t pick_random_pq() {
      return dist(rng);
    }

    [[nodiscard]] std::array<std::size_t, 2> pick_two_random_pqs() {
      std::array<std::size_t, 2> ans{pick_random_pq(), pick_random_pq()};
      while (ans[0] == ans[1]) {
        ans.back() = pick_random_pq();
      }
      return ans;
    }

    std::mt19937 rng;
    std::uniform_int_distribution<std::size_t> dist;
  };

public:
  using PQ = DynamicBinaryHeap<ID, Key, Comparator>;

  struct Handle {
    Handle(std::size_t index, PQ *pq) : _index(index), _pq(pq) {}
    Handle() : _index(0), _pq(nullptr) {}

    Handle(Handle &&) = default;
    Handle &operator=(Handle &&) = default;

    Handle(const Handle &) = delete;
    Handle &operator=(const Handle &) = delete;

    PQ *operator->() {
      KASSERT(_pq != nullptr);

      return _pq;
    }

    operator bool() const {
      return _pq != nullptr;
    }

    [[nodiscard]] std::size_t index() const {
      KASSERT(_pq != nullptr);

      return _index;
    }

  private:
    std::size_t _index;
    PQ *_pq;
  };

  MultiQueue(const int num_threads = 1, const int seed = 0) {
    reset(num_threads, seed);
  }

  Handle lock_pop_pq() {
    auto &token = _token_ets.local();

    // Normal operation: lock two random PQs and return the better one
    for (int attempt = 0; attempt < kNumPopAttempts; ++attempt) {
      const auto [first, second] = token.pick_two_random_pqs();

      if (_pqs[first].empty() && _pqs[second].empty()) {
        continue;
      }

      const std::size_t pq =
          (_pqs[first].empty() || _cmp(_top_keys[first], _top_keys[second])) ? second : first;

      if (!try_lock(pq)) {
        continue;
      }

      if (_pqs[pq].empty()) {
        unlock(pq);
        continue;
      }

      return {pq, &_pqs[pq]};
    }

    // Fallback: test all PQs and return the best one, or indicate termination otherwise
    while (true) {
      Key best_key = Comparator<Key>::kMaxValue;
      std::size_t best_pq = std::numeric_limits<std::size_t>::max();

      for (std::size_t pq = 0; pq < _pqs.size(); ++pq) {
        if (!_pqs[pq].empty() && _cmp(_top_keys[pq], best_key)) {
          best_key = _top_keys[pq];
          best_pq = pq;
        }
      }

      if (best_pq == std::numeric_limits<std::size_t>::max()) {
        return {};
      }

      if (!try_lock(best_pq)) {
        continue;
      }
      if (_pqs[best_pq].empty()) {
        unlock(best_pq);
        continue;
      }

      return {best_pq, &_pqs[best_pq]};
    }
  }

  Handle lock_push_pq() {
    auto &token = _token_ets.local();

    std::size_t pq = 0;
    do {
      pq = token.pick_random_pq();
    } while (try_lock(pq));

    return {pq, &_pqs[pq]};
  }

  void unlock(const Handle handle) {
    KASSERT(!!handle);

    unlock(handle.index());
  }

  void reset() {
    reset(static_cast<int>(_pqs.size() / kNumPQsPerThread), _seed);
  }

  void reset(const int num_threads, const int seed = 0) {
    const std::size_t num_pqs = kNumPQsPerThread * num_threads;

    _seed = seed;

    _pqs.clear();
    _pqs.resize(num_pqs);

    _pq_locks.clear();
    _pq_locks.resize(num_pqs);

    _top_keys.clear();
    _top_keys.assign(num_pqs, Comparator<Key>::kMaxValue);

    _token_ets.clear();
  }

  void free() {
    _pq_locks.clear();
    _pq_locks.shrink_to_fit();

    _pqs.clear();
    _pqs.shrink_to_fit();

    _top_keys.clear();
    _top_keys.shrink_to_fit();

    _token_ets.clear();
  }

private:
  [[nodiscard]] bool is_locked(const std::size_t pq) const {
    KASSERT(pq < _pq_locks.size());

    return __atomic_load_n(&_pq_locks[pq], __ATOMIC_RELAXED);
  }

  bool try_lock(const std::size_t pq) {
    KASSERT(pq < _pq_locks.size());

    std::uint8_t zero = 0u;
    return _pq_locks[pq] == zero &&
           __atomic_compare_exchange_n(
               &_pq_locks[pq], &zero, 1u, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
           );
  }

  void unlock(const std::size_t pq) {
    KASSERT(pq < _pq_locks.size());
    KASSERT(is_locked(pq));

    update_top_key(pq);
    __atomic_store_n(&_pq_locks[pq], 0u, __ATOMIC_RELAXED);
  }

  void update_top_key(const std::size_t pq) {
    KASSERT(is_locked(pq));

    if (!_pqs[pq].empty()) {
      _top_keys[pq] = _pqs[pq].peek_key();
    }
  }

  int _seed;

  std::vector<std::uint8_t> _pq_locks;
  std::vector<PQ> _pqs;
  std::vector<Key> _top_keys;

  tbb::enumerable_thread_specific<Token> _token_ets{[&] {
    return Token(_seed, _pqs.size());
  }};

  Comparator<Key> _cmp;
};

template <typename ID, typename Key>
using MinMultiQueue = MultiQueue<ID, Key, binary_heap::min_heap_comparator>;

template <typename ID, typename Key>
using MaxMultiQueue = MultiQueue<ID, Key, binary_heap::max_heap_comparator>;

} // namespace kaminpar
