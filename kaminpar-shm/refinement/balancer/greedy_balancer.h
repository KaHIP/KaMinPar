/*******************************************************************************
 * Greedy balancing algorithms that uses one thread per overloaded block.
 *
 * @file:   greedy_balancer.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm {
using RelativeGain = double;

struct GreedyBalancerMemoryContext {
  DynamicBinaryMinMaxForest<NodeID, RelativeGain, ScalableVector> pq;
  tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> rating_map;
  tbb::enumerable_thread_specific<std::vector<BlockID>> feasible_target_blocks;
  StaticArray<std::uint8_t> moved_nodes;
  std::vector<BlockWeight> pq_weight;
};

template <typename Graph> class GreedyBalancerImpl : public Refiner {
  SET_DEBUG(false);
  SET_STATISTICS_FROM_GLOBAL();

  struct Statistics {
    EdgeWeight initial_cut;
    EdgeWeight final_cut;
    parallel::Atomic<std::size_t> num_successful_random_moves;
    parallel::Atomic<std::size_t> num_successful_adjacent_moves;
    parallel::Atomic<std::size_t> num_unsuccessful_random_moves;
    parallel::Atomic<std::size_t> num_unsuccessful_adjacent_moves;
    parallel::Atomic<std::size_t> num_moved_border_nodes;
    parallel::Atomic<std::size_t> num_moved_internal_nodes;
    parallel::Atomic<std::size_t> num_pq_reinserts;
    parallel::Atomic<std::size_t> num_overloaded_blocks;
    BlockWeight initial_overload;
    BlockWeight final_overload;
    parallel::Atomic<std::size_t> total_pq_sizes;
    parallel::Atomic<std::size_t> num_feasible_target_block_inits;

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
      STATS << "Greedy Node Balancer:";
      STATS << "  * Changed cut: " << initial_cut << " -> " << final_cut;
      STATS << "  * # overloaded blocks: " << num_overloaded_blocks;
      STATS << "  * # overload change: " << initial_overload << " -> " << final_overload;
      STATS << "  * # moved nodes: " << num_moved_border_nodes + num_moved_internal_nodes << " "
            << "(border nodes: " << num_moved_border_nodes
            << ", internal nodes: " << num_moved_internal_nodes << ")";
      STATS << "  * # successful border node moves: " << num_successful_adjacent_moves << ", "
            << "# unsuccessful border node moves: " << num_unsuccessful_adjacent_moves;
      STATS << "  * # successful random node moves: " << num_successful_random_moves << ", "
            << "# unsuccessful random node moves: " << num_unsuccessful_random_moves;
      STATS << "  * failed moves due to gain changes: " << num_pq_reinserts;
      if (num_overloaded_blocks > 0) {
        STATS << "  * Total initial PQ sizes: " << total_pq_sizes << ", avg "
              << total_pq_sizes / num_overloaded_blocks;
      }
      STATS << "  * Feasible target blocks initialized: " << num_feasible_target_block_inits;
    }
  };

public:
  GreedyBalancerImpl(const Context &ctx)
      : _rating_map([=] { return RatingMap<EdgeWeight, NodeID>{ctx.partition.k}; }) {}

  void setup(GreedyBalancerMemoryContext memory_context) {
    _pq = std::move(memory_context.pq);
    _rating_map = std::move(memory_context.rating_map);
    _feasible_target_blocks = std::move(memory_context.feasible_target_blocks);
    _moved_nodes = std::move(memory_context.moved_nodes);
    _pq_weight = std::move(memory_context.pq_weight);
  }

  GreedyBalancerMemoryContext release() {
    return {
        std::move(_pq),
        std::move(_rating_map),
        std::move(_feasible_target_blocks),
        std::move(_moved_nodes),
        std::move(_pq_weight),
    };
  }

  void initialize([[maybe_unused]] const PartitionedGraph &p_graph) override {}

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override {
    _p_ctx = &p_ctx;
    _p_graph = &p_graph;
    _graph = &concretize<Graph>(p_graph.graph());

    TIMED_SCOPE("Allocation") {
      SCOPED_HEAP_PROFILER("Allocation");
      _moved_nodes.resize(_graph->n());
      _pq.init(_p_graph->k());
      _pq_weight.resize(_p_graph->k());
    };

    _stats.reset();

    const EdgeWeight initial_cut = IFDBG(metrics::edge_cut(*_p_graph));

    init_pq();
    perform_round();

    DBG << "-> Balancer: cut=" << initial_cut << " -> " << metrics::edge_cut(*_p_graph);
    IFSTATS(_stats.print());

    return true;
  }

  void track_moves(NormalSparseGainCache<Graph> *gain_cache) {
    _gain_cache = gain_cache;
  }

private:
  BlockWeight perform_round() {
    IFSTATS(_stats.initial_cut = metrics::edge_cut(*_p_graph));
    IFSTATS(_stats.initial_overload = metrics::total_overload(*_p_graph, *_p_ctx));

    // reset feasible target blocks
    for (auto &blocks : _feasible_target_blocks) {
      blocks.clear();
    }

    tbb::enumerable_thread_specific<BlockWeight> overload_delta;

    START_TIMER("Main loop");
    SCOPED_HEAP_PROFILER("Main loop");
    tbb::parallel_for(static_cast<BlockID>(0), _p_graph->k(), [&](const BlockID from) {
      BlockWeight current_overload = block_overload(from);

      if (current_overload > 0 && _feasible_target_blocks.local().empty()) {
        init_feasible_target_blocks();
        DBG << "Block " << from << " with overload: " << current_overload << ": "
            << _feasible_target_blocks.local().size() << " feasible target blocks and "
            << _pq.size(from) << " nodes in PQ: total weight of PQ is " << _pq_weight[from];
      }

      while (current_overload > 0 && !_pq.empty(from)) {
        KASSERT(
            current_overload ==
            std::max<BlockWeight>(0, _p_graph->block_weight(from) - _p_ctx->max_block_weight(from))
        );

        const NodeID u = _pq.peek_max_id(from);
        const NodeWeight u_weight = _graph->node_weight(u);
        const RelativeGain expected_relative_gain = _pq.peek_max_key(from);
        _pq.pop_max(from);
        _pq_weight[from] -= u_weight;
        KASSERT(_moved_nodes[u] == 1);

        auto [to, actual_relative_gain] = compute_gain(u, from);
        if (expected_relative_gain <= actual_relative_gain) {
          bool moved_node = false;

          if (to == from) { // internal node --> move to random underloaded block
            moved_node = move_to_random_block(u);
            IFSTATS(_stats.num_successful_random_moves += moved_node);
            IFSTATS(_stats.num_unsuccessful_random_moves += (1 - moved_node));
            IFSTATS(++_stats.num_moved_internal_nodes);

            // border node -> move to promising block
          } else if (move_node_if_possible(u, from, to)) {
            moved_node = true;
            IFSTATS(++_stats.num_moved_border_nodes);
            IFSTATS(++_stats.num_successful_adjacent_moves);

            // border node could not be moved -> try again
          } else {
            IFSTATS(++_stats.num_pq_reinserts);
            IFSTATS(++_stats.num_unsuccessful_adjacent_moves);
          }

          if (moved_node) { // update overload if node was moved
            const BlockWeight delta = std::min(current_overload, u_weight);
            current_overload -= delta;
            overload_delta.local() += delta;

            // try to add neighbors of moved node to PQ
            _graph->adjacent_nodes(u, [&](const NodeID v) {
              if (_moved_nodes[v] == 0 && _p_graph->block(v) == from) {
                add_to_pq(from, v);
                _moved_nodes[v] = 1;
              }
            });
          } else if (to != from) {
            // Only re-insert nodes that we tried to move to adjacent blocks
            add_to_pq(from, u, u_weight, actual_relative_gain);
          }
        } else { // gain changed after insertion --> try again with new gain
          add_to_pq(from, u, _graph->node_weight(u), actual_relative_gain);
          IFSTATS(++_stats.num_pq_reinserts);
        }
      }

      KASSERT(
          current_overload ==
          std::max<BlockWeight>(0, _p_graph->block_weight(from) - _p_ctx->max_block_weight(from))
      );
    });
    STOP_TIMER();

    IFSTATS(_stats.final_cut = metrics::edge_cut(*_p_graph));
    IFSTATS(_stats.final_overload = metrics::total_overload(*_p_graph, *_p_ctx));

    const BlockWeight global_overload_delta = overload_delta.combine(std::plus{});
    return global_overload_delta;
  }

  bool add_to_pq(const BlockID b, const NodeID u) {
    KASSERT(b == _p_graph->block(u));

    const auto [to, rel_gain] = compute_gain(u, b);
    return add_to_pq(b, u, _graph->node_weight(u), rel_gain);
  }

  bool add_to_pq(
      const BlockID b, const NodeID u, const NodeWeight u_weight, const RelativeGain rel_gain
  ) {
    KASSERT(u_weight == _graph->node_weight(u));
    KASSERT(b == _p_graph->block(u));

    if (_pq_weight[b] < block_overload(b) || _pq.empty(b) || rel_gain > _pq.peek_min_key(b)) {
      DBG << "Add " << u << " pq weight " << _pq_weight[b] << " rel_gain " << rel_gain;
      _pq.push(b, u, rel_gain);
      _pq_weight[b] += u_weight;

      if (rel_gain > _pq.peek_min_key(b)) {
        const NodeID min_node = _pq.peek_min_id(b);
        const NodeWeight min_weight = _graph->node_weight(min_node);
        if (_pq_weight[b] - min_weight >= block_overload(b)) {
          _pq.pop_min(b);
          _pq_weight[b] -= min_weight;
        }
      }

      return true;
    }

    return false;
  }

  void init_pq() {
    SCOPED_TIMER("Initialize balancer PQ");
    SCOPED_HEAP_PROFILER("Initialize balancer PQ");

    const BlockID k = _p_graph->k();

    using PQs = std::vector<DynamicBinaryMinHeap<NodeID, RelativeGain, ScalableVector>>;
    tbb::enumerable_thread_specific<PQs> local_pq{[&] {
      return PQs(k);
    }};

    using PQWeights = std::vector<NodeWeight>;
    tbb::enumerable_thread_specific<PQWeights> local_pq_weight{[&] {
      return PQWeights(k);
    }};

    // build thread-local PQs: one PQ for each thread and block, each PQ for block
    // b has at most roughly |overload[b]| weight
    START_TIMER("Thread-local");
    tbb::parallel_for(static_cast<NodeID>(0), _graph->n(), [&](const NodeID u) {
      auto &pq = local_pq.local();
      auto &pq_weight = local_pq_weight.local();

      const BlockID b = _p_graph->block(u);
      const BlockWeight overload = block_overload(b);

      if (overload > 0) { // node in overloaded block
        const auto [max_gainer, rel_gain] = compute_gain(u, b);
        const bool need_more_nodes = (pq_weight[b] < overload);
        if (need_more_nodes || pq[b].empty() || rel_gain > pq[b].peek_key()) {
          if (!need_more_nodes) {
            const NodeWeight u_weight = _graph->node_weight(u);
            const NodeWeight min_weight = _graph->node_weight(pq[b].peek_id());
            if (pq_weight[b] + u_weight - min_weight >= overload) {
              pq[b].pop();
            }
          }
          pq[b].push(u, rel_gain);
          _moved_nodes[u] = 1;
        }
      }
    });
    STOP_TIMER();

    // build global PQ: one PQ per block, block-level parallelism
    _pq.clear();

    START_TIMER("Merge thread-local PQs");
    tbb::parallel_for(static_cast<BlockID>(0), k, [&](const BlockID b) {
      IFSTATS(_stats.num_overloaded_blocks += block_overload(b) > 0 ? 1 : 0);

      _pq_weight[b] = 0;

      for (auto &pq : local_pq) {
        for (const auto &[u, rel_gain] : pq[b].elements()) {
          add_to_pq(b, u, _graph->node_weight(u), rel_gain);
        }
      }

      if (!_pq.empty(b)) {
        DBG << "PQ " << b << ": weight=" << _pq_weight[b] << ", " << _pq.peek_min_key(b)
            << " < key < " << _pq.peek_max_key(b);
      } else {
        DBG << "PQ " << b << ": empty";
      }
    });
    STOP_TIMER();

    _stats.total_pq_sizes = _pq.size();
  }

  [[nodiscard]] std::pair<BlockID, RelativeGain>
  compute_gain(const NodeID u, const BlockID u_block) const {
    const NodeWeight u_weight = _graph->node_weight(u);
    BlockID max_gainer = u_block;
    EdgeWeight max_external_gain = 0;
    EdgeWeight internal_degree = 0;

    auto action = [&](auto &map) {
      // compute external degree to each adjacent block that can take u without
      // becoming overloaded
      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeID w) {
        const BlockID v_block = _p_graph->block(v);
        if (u_block != v_block &&
            _p_graph->block_weight(v_block) + u_weight <= _p_ctx->max_block_weight(v_block)) {
          map[v_block] += w;
        } else if (u_block == v_block) {
          internal_degree += w;
        }
      });

      // select neighbor that maximizes gain
      Random &rand = Random::instance();
      for (const auto [block, gain] : map.entries()) {
        if (gain > max_external_gain || (gain == max_external_gain && rand.random_bool())) {
          max_gainer = block;
          max_external_gain = gain;
        }
      }
      map.clear();
    };

    _rating_map.local().execute(_graph->degree(u), action);

    // compute absolute and relative gain based on internal degree / external gain
    const EdgeWeight gain = max_external_gain - internal_degree;
    const RelativeGain relative_gain = compute_relative_gain(gain, u_weight);
    return {max_gainer, relative_gain};
  }

  bool move_node_if_possible(const NodeID u, const BlockID from, const BlockID to) {
    if (_p_graph->move(u, from, to, _p_ctx->max_block_weight(to))) {
      if (_gain_cache != nullptr) {
        _gain_cache->move(u, from, to);
      }
      return true;
    }

    return false;
  }

  bool move_to_random_block(const NodeID u) {
    auto &feasible_target_blocks = _feasible_target_blocks.local();
    const BlockID u_block = _p_graph->block(u);

    while (!feasible_target_blocks.empty()) {
      // get random block from feasible block list
      const std::size_t n = feasible_target_blocks.size();
      const std::size_t i = Random::instance().random_index(0, n);
      const BlockID b = feasible_target_blocks[i];
      KASSERT(u_block != b);

      // try to move node to that block, if possible, operation succeeded
      if (move_node_if_possible(u, u_block, b)) {
        return true;
      }

      // loop terminated without return, hence moving u to b failed --> we no
      // longer consider b to be a feasible target block and remove it from the
      // list
      std::swap(feasible_target_blocks[i], feasible_target_blocks.back());
      feasible_target_blocks.pop_back();
    }

    // there are no more feasible target blocks -> operation failed
    return false;
  }

  void init_feasible_target_blocks() {
    IFSTATS(++_stats.num_feasible_target_block_inits);

    auto &blocks = _feasible_target_blocks.local();
    blocks.clear();
    for (const BlockID b : _p_graph->blocks()) {
      if (_p_graph->block_weight(b) < _p_ctx->perfectly_balanced_block_weight(b)) {
        blocks.push_back(b);
      }
    }
  }

  [[nodiscard]] inline BlockWeight block_overload(const BlockID b) const {
    static_assert(
        std::numeric_limits<BlockWeight>::is_signed,
        "This must be changed when using an unsigned data type for "
        "block weights!"
    );

    return std::max<BlockWeight>(0, _p_graph->block_weight(b) - _p_ctx->max_block_weight(b));
  }

  [[nodiscard]] static inline RelativeGain
  compute_relative_gain(const EdgeWeight absolute_gain, const NodeWeight weight) {
    if (absolute_gain >= 0) {
      return absolute_gain * weight;
    } else {
      return static_cast<RelativeGain>(absolute_gain) / weight;
    }
  }

  const PartitionContext *_p_ctx;
  PartitionedGraph *_p_graph;
  const Graph *_graph;

  DynamicBinaryMinMaxForest<NodeID, RelativeGain, ScalableVector> _pq;
  mutable tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> _rating_map;
  tbb::enumerable_thread_specific<std::vector<BlockID>> _feasible_target_blocks;
  StaticArray<std::uint8_t> _moved_nodes;
  std::vector<BlockWeight> _pq_weight;

  Statistics _stats;

  NormalSparseGainCache<Graph> *_gain_cache = nullptr;
};

class GreedyBalancer : public Refiner {
  using GreedyBalancerCSRImpl = GreedyBalancerImpl<CSRGraph>;
  using GreedyBalancerCompressedImpl = GreedyBalancerImpl<CompressedGraph>;

public:
  GreedyBalancer(const Context &ctx);
  ~GreedyBalancer() override;

  GreedyBalancer &operator=(const GreedyBalancer &) = delete;
  GreedyBalancer(const GreedyBalancer &) = delete;

  GreedyBalancer &operator=(GreedyBalancer &&) = default;
  GreedyBalancer(GreedyBalancer &&) noexcept = default;

  [[nodiscard]] std::string name() const final;

  void initialize(const PartitionedGraph &p_graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

private:
  std::unique_ptr<GreedyBalancerCSRImpl> _csr_impl;
  std::unique_ptr<GreedyBalancerCompressedImpl> _compressed_impl;

  GreedyBalancerMemoryContext _memory_context;
};

} // namespace kaminpar::shm
