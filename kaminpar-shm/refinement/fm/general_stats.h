/*******************************************************************************
 * @file:   general_stats.h
 * @author: Daniel Seemaier
 * @date:   30.07.2024
 ******************************************************************************/
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace kaminpar::shm::fm {

enum class Statistic {
  NUM_TOUCHED_NODES,
  NUM_COMMITTED_MOVES,
  NUM_DISCARDED_MOVES,
  NUM_RECOMPUTED_GAINS,
  NUM_BATCHES,
  NUM_PQ_INSERTS,
  NUM_PQ_UPDATES,
  NUM_PQ_POPS,
  N,
};

class IterationStatistics {
  constexpr static std::size_t kNumberOfStatistics = static_cast<std::size_t>(Statistic::N);

public:
  IterationStatistics();
  IterationStatistics(IterationStatistics &&);

  void operator()(const Statistic stat, std::int64_t by = 1);

  [[nodiscard]] std::int64_t get(const Statistic stat) const;

  IterationStatistics &operator+=(const IterationStatistics &other);

private:
  std::array<std::atomic<std::int64_t>, kNumberOfStatistics> _stats;
};

class GlobalStatistics {
public:
  GlobalStatistics();

  void add(const IterationStatistics &stats);
  void next_iteration();
  void reset();
  void print();

private:
  std::vector<IterationStatistics> iteration_stats;
};

} // namespace kaminpar::shm::fm
