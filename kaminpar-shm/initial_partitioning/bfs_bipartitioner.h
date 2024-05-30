/*******************************************************************************
 * BFS based initial bipartitioner.
 *
 * @file:   bfs_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <array>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/initial_partitioning/bipartitioner.h"
#include "kaminpar-shm/initial_partitioning/seed_node_utils.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/queue.h"

namespace kaminpar::shm::ip {
namespace bfs {
using Queues = std::array<Queue<NodeID>, 2>;

/*! Always selects the inactive block, i.e., switches blocks after each step. */
struct alternating {
  BlockID
  operator()(const BlockID active_block, const Bipartitioner::BlockWeights &, const PartitionContext &, const Queues &) {
    return 1 - active_block;
  }
};

/*! Always selects the block with the smaller weight. */
struct lighter {
  BlockID
  operator()(const BlockID, const Bipartitioner::BlockWeights &block_weights, const PartitionContext &, const Queues &) {
    return (block_weights[0] < block_weights[1]) ? 0 : 1;
  }
};

/*! Selects the first block until it has more than half weight. */
struct sequential {
  BlockID
  operator()(const BlockID, const Bipartitioner::BlockWeights &block_weights, const PartitionContext &context, const Queues &) {
    return (block_weights[0] < context.block_weights.perfectly_balanced(0)) ? 0 : 1;
  }
};

/*! Always selects the block with the longer queue. */
struct longer_queue {
  BlockID operator()(
      const BlockID,
      const Bipartitioner::BlockWeights &,
      const PartitionContext &,
      const Queues &queues
  ) {
    return (queues[0].size() < queues[1].size()) ? 1 : 0;
  }
};

/*! Always selects the block with the shorter queue. */
struct shorter_queue {
  BlockID operator()(
      const BlockID,
      const Bipartitioner::BlockWeights &,
      const PartitionContext &,
      const Queues &queues
  ) {
    return (queues[0].size() < queues[1].size()) ? 0 : 1;
  }
};

/*!
 * Grows two blocks starting from pseudo peripheral nodes using a breath first
 * search.
 *
 * In each step, a node is assigned to one of the two blocks. A block selection
 * strategy is used to decide when to switch between blocks.
 *
 * @tparam block_selection_strategy Invoked after each step to select the active
 * block in the next step.
 * @tparam seed_node If specified, a fixed seed node from the search for pseudo
 * peripheral nodes is started. If not specified, a random node is used instead.
 */
template <typename block_selection_strategy, NodeID seed_node = kInvalidNodeID>
class BfsBipartitioner : public Bipartitioner {
  static constexpr std::size_t kMarkAssigned = 2;

public:
  BfsBipartitioner(const InitialPartitioningContext &i_ctx)
      : Bipartitioner(i_ctx),
        _num_seed_iterations(i_ctx.num_seed_iterations) {}

  void init(const CSRGraph &graph, const PartitionContext &p_ctx) override {
    Bipartitioner::init(graph, p_ctx);

    if (_marker.capacity() < _graph->n()) {
      _marker.resize(_graph->n());
    }
    if (_queues[0].capacity() < _graph->n()) {
      _queues[0].resize(_graph->n());
    }
    if (_queues[1].capacity() < _graph->n()) {
      _queues[1].resize(_graph->n());
    }
  }

protected:
  void bipartition_impl() override {
    const auto [start_a, start_b] = ip::find_far_away_nodes(*_graph, _num_seed_iterations);

    _marker.reset();
    for (const auto i : {0, 1}) {
      _queues[i].clear();
    }

    _queues[0].push_tail(start_a);
    _queues[1].push_tail(start_b);
    _marker.set<true>(start_a, 0);
    _marker.set<true>(start_b, 1);

    BlockID active = 0;

    while (_marker.first_unmarked_element(kMarkAssigned) < _graph->n()) {
      if (__builtin_expect(_queues[active].empty(), 0)) {
        const auto first_unassigned_node =
            static_cast<NodeID>(_marker.first_unmarked_element(kMarkAssigned));
        // did not fit into active_block -> wait for the other block to catch up
        if (_marker.get(first_unassigned_node, active)) {
          active = 1 - active;
          continue;
        }
        _queues[active].push_tail(first_unassigned_node);
        _marker.set(first_unassigned_node, active);
      }

      const NodeID u = _queues[active].head();
      _queues[active].pop_head();

      // The node could have been assigned between the time it was put into the
      // queue and now, hence we must check this here
      if (__builtin_expect(!_marker.get(u, kMarkAssigned), 1)) {
        // If the balance constraint does not allow us to put this node into the
        // active block, we switch to the lighter block instead
        // --------------------------------------------------
        // This version seems to perform worse in terms of balance
        //
        //        const bool balanced = (_block_weights[active] +
        //        _graph.node_weight(u) <= _p_ctx.max_block_weight(active)); if
        //        (!balanced) {
        //          active = _block_weights[1 - active] < _block_weights[active]
        //          ? 1 - active : active;
        //        }
        //
        // than this version:
        const NodeWeight weight = _block_weights[active];
        const bool assignment_allowed =
            (weight + _graph->node_weight(u) <= _p_ctx->block_weights.max(active));
        active = assignment_allowed * active + (1 - assignment_allowed) * (1 - active);

        set_block(u, active);
        _marker.set<true>(u, kMarkAssigned);

        for (const NodeID v : _graph->adjacent_nodes(u)) {
          if (_marker.get(v, kMarkAssigned) || _marker.get(v, active)) {
            continue;
          }
          _queues[active].push_tail(v);
          _marker.set(v, active);
        }
      }

      active = _block_selection_strategy(active, _block_weights, *_p_ctx, _queues);
    }
  }

private:
  std::array<Queue<NodeID>, 2> _queues{};
  Marker<3> _marker{};

  const std::size_t _num_seed_iterations;

  block_selection_strategy _block_selection_strategy{};
};
} // namespace bfs

extern template class bfs::BfsBipartitioner<bfs::alternating>;
extern template class bfs::BfsBipartitioner<bfs::lighter>;
extern template class bfs::BfsBipartitioner<bfs::sequential>;
extern template class bfs::BfsBipartitioner<bfs::longer_queue>;
extern template class bfs::BfsBipartitioner<bfs::shorter_queue>;

using AlternatingBfsBipartitioner = bfs::BfsBipartitioner<bfs::alternating>;
using LighterBlockBfsBipartitioner = bfs::BfsBipartitioner<bfs::lighter>;
using SequentialBfsBipartitioner = bfs::BfsBipartitioner<bfs::sequential>;
using LongerQueueBfsBipartitioner = bfs::BfsBipartitioner<bfs::longer_queue>;
using ShorterQueueBfsBipartitioner = bfs::BfsBipartitioner<bfs::shorter_queue>;
} // namespace kaminpar::shm::ip
