/*******************************************************************************
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Parallel k-way FM refinement algorithm.
 ******************************************************************************/
#include "kaminpar/refinement/fm_refiner.h"

#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/definitions.h"

#include "common/parallel/atomic.h"

namespace kaminpar::shm {
FMRefiner::FMRefiner(const Context &ctx)
    : _locked(ctx.partition.n),
      _pq_ets([&] { return BinaryMinHeap<EdgeWeight>(ctx.partition.n); }),
      _target_blocks(ctx.partition.n) {}

void FMRefiner::initialize(const Graph &graph) { ((void)graph); }

bool FMRefiner::refine(PartitionedGraph &p_graph,
                       const PartitionContext &p_ctx) {
  _p_graph = &p_graph;
  _p_ctx = &p_ctx;

  p_graph.pfor_nodes([&](NodeID u) { _locked[u] = 0; });
  _gain_cache.reinit(_p_graph);
  init_border_nodes();

  // @todo move to Context
  const NodeID num_seed_nodes = 25;

  tbb::parallel_for<int>(
      0, tbb::this_task_arena::max_concurrency(), [&, this](int) {
        auto &pq = _pq_ets.local();
        DeltaPartitionedGraph d_graph(_p_graph);
        std::vector<NodeID> touched_nodes;

        pq.clear();
        d_graph.clear();

        poll_border_nodes(num_seed_nodes, [&](const NodeID u) {
          const auto [block, gain] =
              _gain_cache.best_gain(u, _p_ctx->block_weights);
          pq.push(u, gain);
          _target_blocks[u] = block;
          touched_nodes.push_back(u);
        });

        EdgeWeight total_gain = 0;

        while (!pq.empty()) {
          const NodeID node = pq.peek_id();
          const EdgeWeight gain = pq.peek_key();
          const BlockID to = _target_blocks[node];
          const NodeWeight node_weight = _p_graph->node_weight(node);
          pq.pop();

          // If gain is no longer valid, re-insert the node
          if (gain != _gain_cache.gain_to(node, to)) {
            const auto [block, gain] =
                _gain_cache.best_gain(node, _p_ctx->block_weights);
            pq.push(node, gain);
            _target_blocks[node] = block;
          }

          if (d_graph.block_weight(to) + node_weight <=
              _p_ctx->block_weights.max(to)) {
            total_gain += gain;
            d_graph.set_block(node, to);

            if (total_gain > 0) {
              for (const auto &[u, b] : d_graph.delta()) {
                _gain_cache.move_node(u, _p_graph->block(u), b);
                _p_graph->set_block(u, b);
              }
              touched_nodes.clear();
            }

            for (const auto &[e, v] : _p_graph->neighbors(node)) {
              if (lock_node(v)) {

                const auto [block, gain] =
                    _gain_cache.best_gain(v, _p_ctx->block_weights);
                pq.push(v, gain);
                _target_blocks[v] = block;
                touched_nodes.push_back(v);
              }
            }
          }
        }

        for (const NodeID u : touched_nodes) {
          unlock_node(u);
        }
      });

  return false;
}

void FMRefiner::init_border_nodes() {
  _border_nodes.clear();
  _p_graph->pfor_nodes([&](const NodeID u) {
    if (_gain_cache.is_border_node(u)) {
      _border_nodes.push_back(u);
    }
    _locked[u] = 0;
  });
}

bool FMRefiner::lock_node(const NodeID u) {
  std::uint8_t free = 0;
  return __atomic_compare_exchange_n(&_locked[u], &free, 1, false,
                                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

void FMRefiner::unlock_node(const NodeID u) {
  _locked[u] = 0;
  if (_gain_cache.is_border_node(u)) {
    _border_nodes.push_back(u);
  }
}
} // namespace kaminpar::shm
