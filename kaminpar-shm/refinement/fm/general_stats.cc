/*******************************************************************************
 * @file:   general_stats.cc
 * @author: Daniel Seemaier
 * @date:   30.07.2024
 ******************************************************************************/
#include "kaminpar-shm/refinement/fm/general_stats.h"

#include <cstdint>
#include <sstream>

#include "kaminpar-common/logger.h"

namespace kaminpar::shm::fm {

IterationStatistics::IterationStatistics() : _stats{} {}

IterationStatistics::IterationStatistics(IterationStatistics &&other) {
  *this += other;
}

void IterationStatistics::operator()(const Statistic stat, const std::int64_t by) {
  _stats[static_cast<std::size_t>(stat)].fetch_add(by, std::memory_order::relaxed);
}

std::int64_t IterationStatistics::get(const Statistic stat) const {
  return _stats[static_cast<std::size_t>(stat)].load(std::memory_order::relaxed);
}

IterationStatistics &IterationStatistics::operator+=(const IterationStatistics &other) {
  for (std::size_t i = 0; i < kNumberOfStatistics; ++i) {
    _stats[i].fetch_add(other.get(static_cast<Statistic>(i)), std::memory_order::relaxed);
  }

  return *this;
}

GlobalStatistics::GlobalStatistics() {
  next_iteration();
}

void GlobalStatistics::add(const IterationStatistics &stats) {
  iteration_stats.back() += stats;
}

void GlobalStatistics::next_iteration() {
  iteration_stats.emplace_back();
}

void GlobalStatistics::reset() {
  iteration_stats.clear();
  next_iteration();
}

void GlobalStatistics::print() {
  std::stringstream ss;

  ss << "component=fm ";
  ss << "iterations=" << iteration_stats.size() << " ";

  for (std::size_t i = 0; i < iteration_stats.size(); ++i) {
    const auto &stats = iteration_stats[i];

    const auto num_batches = stats.get(Statistic::NUM_BATCHES);
    const auto num_touched_nodes = stats.get(Statistic::NUM_TOUCHED_NODES);
    const auto num_touched_nodes_per_batch =
        (num_batches > 0) ? 1.0 * num_touched_nodes / num_batches : 0;
    const auto num_committed_moves = stats.get(Statistic::NUM_COMMITTED_MOVES);
    const auto num_discarded_moves = stats.get(Statistic::NUM_DISCARDED_MOVES);
    const auto fraction_discarded =
        1.0 * num_discarded_moves / (num_committed_moves + num_discarded_moves);
    const auto num_recomputed_gains = stats.get(Statistic::NUM_RECOMPUTED_GAINS);
    const auto num_pq_inserts = stats.get(Statistic::NUM_PQ_INSERTS);
    const auto num_pq_updates = stats.get(Statistic::NUM_PQ_UPDATES);
    const auto num_pq_pops = stats.get(Statistic::NUM_PQ_POPS);

    ss << "num_batches(" << i << ")=" << num_batches << " ";
    ss << "num_touched_nodes(" << i << ")=" << num_touched_nodes << " ";
    ss << "num_touched_nodes_per_batch(" << i << ")=" << num_touched_nodes_per_batch << " ";
    ss << "num_committed_moves(" << i << ")=" << num_committed_moves << " ";
    ss << "num_discarded_moves(" << i << ")=" << num_discarded_moves << " ";
    ss << "fraction_discarded(" << i << ")=" << fraction_discarded << " ";
    ss << "num_recomputed_gains(" << i << ")=" << num_recomputed_gains << " ";
    ss << "num_pq_inserts(" << i << ")=" << num_pq_inserts << " ";
    ss << "num_pq_updates(" << i << ")=" << num_pq_updates << " ";
    ss << "num_pq_pops(" << i << ")=" << num_pq_pops << " ";
  }

  LOG_STATS << ss.str();
}

} // namespace kaminpar::shm::fm
