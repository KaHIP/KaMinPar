/*******************************************************************************
 * @file:   loops.h
 * @author: Daniel Seemaier
 * @date:   30.03.2022
 * @brief:  Helpers for parallel loops.
 ******************************************************************************/
#pragma once

#include <type_traits>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

namespace kaminpar::parallel {
/*!
 * @param buffers Vector of buffers of elements.
 * @param lambda Invoked on each element, in parallel.
 */
template <typename Buffer, typename Lambda> void chunked_for(Buffer &buffers, Lambda &&lambda) {
  std::size_t total_size = 0;
  for (const auto &buffer : buffers) {
    total_size += buffer.size();
  }

  constexpr bool invocable_with_chunk_id =
      std::is_invocable_v<decltype(lambda), decltype(buffers[0][0]), int>;

  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, total_size), [&](const auto r) {
    std::size_t cur = r.begin();
    std::size_t offset = 0;
    std::size_t current_buf = 0;
    std::size_t cur_size = buffers[current_buf].size();

    // find first buffer for our range
    while (offset + cur_size < cur) {
      offset += cur_size;
      ++current_buf;
      KASSERT(current_buf < buffers.size());
      cur_size = buffers[current_buf].size();
    }

    // iterate elements
    while (cur != r.end()) {
      while (cur - offset >= cur_size) {
        KASSERT(current_buf < buffers.size());
        offset += buffers[current_buf++].size();
        cur_size = buffers[current_buf].size();
      }
      KASSERT(current_buf < buffers.size());
      KASSERT(cur_size == buffers[current_buf].size());
      KASSERT(cur - offset < buffers[current_buf].size());
      if constexpr (invocable_with_chunk_id) {
        lambda(buffers[current_buf][cur - offset], current_buf);
      } else {
        lambda(buffers[current_buf][cur - offset]);
      }
      ++cur;
    }
  });
}

/*!
 * Iterate over from..to in parallel, where from..from+x are processed by CPU 0,
 * from+x..from+2x by CPU 1 etc. The number of CPUs is determined by the maximum
 * concurrency of the current TBB task arena.
 *
 * @tparam Index
 * @tparam Lambda Must take a start and end index and the CPU id.
 * @param from
 * @param to
 * @param lambda Called once for each CPU (in parallel): first element, first
 * invalid element, CPU id
 */
template <typename Index, typename Lambda>
void deterministic_for(const Index from, const Index to, Lambda &&lambda) {
  static_assert(std::is_invocable_v<Lambda, Index, Index, int>);

  const Index n = to - from;
  const int p = std::min<int>(tbb::this_task_arena::max_concurrency(), n);

  tbb::parallel_for(static_cast<int>(0), p, [&](const int cpu) {
    const Index chunk = n / p;
    const Index rem = n % p;
    const Index cpu_from = cpu * chunk + std::min(cpu, static_cast<int>(rem));
    const Index cpu_to = cpu_from + ((cpu < static_cast<int>(rem)) ? chunk + 1 : chunk);

    lambda(from + cpu_from, from + cpu_to, cpu);
  });
}
} // namespace kaminpar::parallel
