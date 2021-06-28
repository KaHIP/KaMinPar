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

#include "context.h"
#include "datastructure/binary_heap.h"
#include "datastructure/fast_reset_array.h"
#include "datastructure/graph.h"
#include "datastructure/rating_map.h"
#include "datastructure/marker.h"
#include "refinement/i_balancer.h"
#include "utility/metrics.h"
#include "utility/random.h"
#include "utility/timer.h"

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>

namespace kaminpar {
class ParallelBalancer : public Balancer {
  SET_DEBUG(false);
  SET_STATISTICS(false);
  SET_OUTPUT(false);

public:
  struct Statistics {
    EdgeWeight initial_cut;
    EdgeWeight final_cut;
    parallel::IntegralAtomicWrapper<std::size_t> num_successful_random_moves;
    parallel::IntegralAtomicWrapper<std::size_t> num_successful_adjacent_moves;
    parallel::IntegralAtomicWrapper<std::size_t> num_unsuccessful_random_moves;
    parallel::IntegralAtomicWrapper<std::size_t> num_unsuccessful_adjacent_moves;
    parallel::IntegralAtomicWrapper<std::size_t> num_moved_border_nodes;
    parallel::IntegralAtomicWrapper<std::size_t> num_moved_internal_nodes;
    parallel::IntegralAtomicWrapper<std::size_t> num_pq_reinserts;
    parallel::IntegralAtomicWrapper<std::size_t> num_overloaded_blocks;
    BlockWeight initial_overload;
    BlockWeight final_overload;
    parallel::IntegralAtomicWrapper<std::size_t> total_pq_sizes;
    parallel::IntegralAtomicWrapper<std::size_t> num_feasible_target_block_inits;

    void reset() {
      initial_cut = 0;
      final_cut = 0;
      num_successful_random_moves = 0;
      num_successful_adjacent_moves = 0;
      num_unsuccessful_random_moves = 0;
      num_unsuccessful_adjacent_moves = 0;
      num_moved_border_nodes = 0;
      num_moved_internal_nodes = 0;
      num_pq_reinserts = 0;
      num_overloaded_blocks = 0;
      initial_overload = 0;
      final_overload = 0;
      total_pq_sizes = 0;
      num_feasible_target_block_inits = 0;
    }

    void print() {
      LOG << logger::CYAN << "Balancer Statistics:";
      LOG << logger::CYAN << "* Changed cut: " << C(initial_cut, final_cut);
      LOG << logger::CYAN << "* # overloaded blocks: " << num_overloaded_blocks;
      LOG << logger::CYAN << "* # overload change: " << C(initial_overload, final_overload);
      LOG << logger::CYAN << "* # moved nodes: " << num_moved_border_nodes + num_moved_internal_nodes << " "
          << "(border nodes: " << num_moved_border_nodes << ", internal nodes: " << num_moved_internal_nodes << ")";
      LOG << logger::CYAN << "* # successful border node moves: " << num_successful_adjacent_moves << ", "
          << "# unsuccessful border node moves: " << num_unsuccessful_adjacent_moves;
      LOG << logger::CYAN << "* # successful random node moves: " << num_successful_random_moves << ", "
          << "# unsuccessful random node moves: " << num_unsuccessful_random_moves;
      LOG << logger::CYAN << "* failed moves due to gain changes: " << num_pq_reinserts;
      if (num_overloaded_blocks > 0) {
        LOG << logger::CYAN << "* total initial PQ sizes: " << total_pq_sizes << ", avg "
            << total_pq_sizes / num_overloaded_blocks;
      }
      LOG << logger::CYAN << "* feasible target blocks initialized: " << num_feasible_target_block_inits;
    }
  };

  ParallelBalancer(const Graph &graph, const BlockID max_k, const RefinementContext &)
      : _max_k{max_k},
        _pq{graph.n(), max_k},
        _marker{graph.n()},
        _pq_weight(max_k) {}

  ParallelBalancer(const PartitionedGraph &) = delete;
  ParallelBalancer(ParallelBalancer &&) noexcept = default;
  ParallelBalancer &operator=(const ParallelBalancer &) = delete;
  ParallelBalancer &operator=(ParallelBalancer &&) = delete;

  void initialize(const PartitionedGraph &p_graph) final;
  bool balance(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  BlockWeight perform_round();

  bool move_node_if_possible(const NodeID u, const BlockID from, const BlockID to);

  bool move_to_random_block(const NodeID u);

  void init_pq();

  bool add_to_pq(const BlockID b, const NodeID u);

  bool add_to_pq(const BlockID b, const NodeID u, const NodeWeight u_weight, const double rel_gain);

  [[nodiscard]] std::pair<BlockID, double> compute_gain(const NodeID u, const BlockID u_block) const;

  void init_feasible_target_blocks();

  [[nodiscard]] static inline double compute_relative_gain(const EdgeWeight absolute_gain, const NodeWeight weight) {
    if (absolute_gain >= 0) {
      return absolute_gain * weight;
    } else {
      return 1.0 * absolute_gain / weight;
    }
  }

  inline BlockWeight block_overload(const BlockID b) const {
    static_assert(std::numeric_limits<BlockWeight>::is_signed,
                  "This must be changed when using an unsigned data type for block weights!");
    return std::max(0, _p_graph->block_weight(b) - _p_ctx->max_block_weight(b));
  }

  const BlockID _max_k;

  PartitionedGraph *_p_graph;
  const PartitionContext *_p_ctx;

  DynamicBinaryMinMaxForest<NodeID, double> _pq;
  mutable tbb::enumerable_thread_specific<RatingMap<EdgeWeight>> _rating_map{
      [&] { return RatingMap<EdgeWeight>{_max_k}; }};
  tbb::enumerable_thread_specific<std::vector<BlockID>> _feasible_target_blocks;
  Marker<> _marker;
  std::vector<BlockWeight> _pq_weight;

  Statistics _stats;
};
} // namespace kaminpar
