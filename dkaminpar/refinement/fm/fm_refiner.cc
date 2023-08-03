/*******************************************************************************
 * Distributed FM refiner.
 *
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   11.09.2022
 ******************************************************************************/
#include "dkaminpar/refinement/fm/fm_refiner.h"

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "dkaminpar/algorithms/independent_set.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/datastructures/growt.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/bfs_extractor.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/refinement/fm/move_conflict_resolver.h"

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/refinement/fm_refiner.h"
#include "kaminpar/refinement/stopping_policies.h"

#include "common/datastructures/binary_heap.h"
#include "common/datastructures/rating_map.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
FMRefinerFactory::FMRefinerFactory(const Context &ctx) : _ctx(ctx) {}

std::unique_ptr<GlobalRefiner>
FMRefinerFactory::create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return std::make_unique<FMRefiner>(_ctx, p_graph, p_ctx);
}

FMRefiner::FMRefiner(
    const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _ctx(ctx),
      _fm_ctx(ctx.refinement.fm),
      _p_graph(p_graph),
      _p_ctx(p_ctx),
      _balancer_factory(factory::create_refiner(_ctx, _fm_ctx.balancing_algorithm)) {}

void FMRefiner::initialize() {}

bool FMRefiner::refine() {
  SCOPED_TIMER("FM");

  auto balancer = _balancer_factory->create(_p_graph, _p_ctx);
  balancer->initialize();

  for (std::size_t global_round = 0; global_round < _fm_ctx.num_global_iterations; ++global_round) {
    const auto seed_nodes = graph::find_independent_border_set(_p_graph, global_round);

    graph::BfsExtractor bfs_extractor(_p_graph.graph());
    bfs_extractor.initialize(_p_graph);
    bfs_extractor.set_max_hops(_fm_ctx.max_hops);
    bfs_extractor.set_max_radius(_fm_ctx.max_radius);
    auto extraction_result = bfs_extractor.extract(seed_nodes);

    shm::Graph *b_graph = extraction_result.graph.get();
    shm::PartitionedGraph *bp_graph = extraction_result.p_graph.get();
    auto &node_mapping = extraction_result.node_mapping;

    DBG << "BFS extraction result: n=" << b_graph->n() << ", m=" << b_graph->m();
    KASSERT(
        shm::validate_graph(
            *b_graph, false, _p_graph.k()
        ), // @todo why is the graph (outside the fixed vertices) not undirected?
        "BFS extractor returned invalid graph data structure",
        assert::heavy
    );

    // Overwrite the block weights in the batch graphs with block weights of the global graph
    for (const BlockID block : _p_graph.blocks()) {
      bp_graph->set_block_weight(block, _p_graph.block_weight(block));
    }

    START_TIMER("Build reverse node mapping");
    growt::StaticGhostNodeMapping reverse_node_mapping(extraction_result.node_mapping.size());
    tbb::parallel_for<std::size_t>(
        0,
        extraction_result.node_mapping.size(),
        [&](const std::size_t i) {
          reverse_node_mapping.insert(extraction_result.node_mapping[i] + 1, i);
        }
    );
    mpi::barrier(_p_graph.communicator());
    STOP_TIMER();

    shm::PartitionContext shm_p_ctx;
    shm_p_ctx.epsilon = _p_ctx.epsilon;
    shm_p_ctx.k = _p_ctx.k;
    shm_p_ctx.n = asserting_cast<NodeID>(b_graph->n());
    shm_p_ctx.m = asserting_cast<EdgeID>(b_graph->m());
    shm_p_ctx.total_node_weight =
        asserting_cast<NodeWeight>(_p_ctx.graph->global_total_node_weight);
    shm_p_ctx.total_edge_weight =
        asserting_cast<EdgeWeight>(_p_ctx.graph->global_total_edge_weight);
    shm_p_ctx.max_node_weight = asserting_cast<NodeWeight>(_p_ctx.graph->global_max_node_weight);
    shm_p_ctx.setup_block_weights();

    shm::fm::SharedData shared(b_graph->n(), _p_ctx.k);
    tbb::concurrent_vector<NodeID> b_seed_nodes;
    tbb::parallel_for<std::size_t>(0, seed_nodes.size(), [&](const std::size_t i) {
      const NodeID local_seed_node = seed_nodes[i];
      const GlobalNodeID global_seed_node = _p_graph.local_to_global_node(local_seed_node);
      KASSERT(reverse_node_mapping.find(global_seed_node + 1) != reverse_node_mapping.end());
      b_seed_nodes.push_back((*reverse_node_mapping.find(global_seed_node + 1)).second);
    });

    shared.border_nodes.init_precomputed(*bp_graph, b_seed_nodes);
    shared.border_nodes.shuffle();

    shared.gain_cache.initialize(*bp_graph);

    // Mark pseudo-block nodes as already moved
    for (const BlockID block : _p_graph.blocks()) {
      shared.node_tracker.set(bp_graph->n() - block - 1, shm::fm::NodeTracker::MOVED_GLOBALLY);
    }

    shm::KwayFMRefinementContext shm_fm_ctx{
        .num_seed_nodes = 1,
        .alpha = _fm_ctx.alpha,
        .num_iterations = 1,
        .unlock_seed_nodes = false,
        .use_exact_abortion_threshold = false,
        .abortion_threshold = 0.999,
    };

    // Create thread-local workers numbered 1..P
    std::atomic<int> next_id = 0;
    tbb::enumerable_thread_specific<shm::LocalizedFMRefiner> worker_ets([&] {
      // It is important that worker IDs start at 1, otherwise the node
      // tracker won't work
      shm::LocalizedFMRefiner worker(++next_id, shm_p_ctx, shm_fm_ctx, *bp_graph, shared);

      // This allows access to the moves that were applied to shared bp_graph partition
      worker.enable_move_recording();

      return worker;
    });

    std::vector<GlobalMove> move_sets;

    for (std::size_t local_round = 0; local_round < _fm_ctx.num_local_iterations; ++local_round) {
      DBG << "Starting FM round " << global_round << "/" << local_round;

      shm::LocalizedFMRefiner &worker = worker_ets.local();
      while (shared.border_nodes.has_more()) {
        const NodeID seed_node = shared.border_nodes.get();
        DBG << "Running with seed_node=" << seed_node;

        const EdgeWeight gain = worker.run_batch();

        auto moves = worker.take_applied_moves();
        if (!moves.empty()) {
          const NodeID group = node_mapping[seed_node];
          for (const auto &[node, from] : moves) {
            move_sets.push_back(GlobalMove{
                .node = node_mapping[node],
                .group = seed_node,
                .weight = static_cast<NodeWeight>(bp_graph->node_weight(node)),
                .gain = gain,
                .from = from,
                .to = bp_graph->block(node),
            });
          }

          if (_fm_ctx.revert_local_moves_after_batch) {
            for (const auto &[node, from] : moves) {
              const BlockID to = bp_graph->block(node);
              bp_graph->set_block(node, from);
              shared.gain_cache.move(*bp_graph, node, to, from);
            }
          }
        }
      }

      mpi::barrier(_p_graph.communicator());

      // Resolve global move conflicts
      START_TIMER("Move conflict resolution");
      auto global_move_buffer =
          broadcast_and_resolve_global_moves(move_sets, _p_graph.communicator());
      STOP_TIMER();

      mpi::barrier(_p_graph.communicator());

      // @todo optimize this
      for (const auto &moved_node : move_sets) {
        if (is_invalid_id(moved_node.node)) {
          global_move_buffer.push_back(moved_node);
        }
      }

      // Apply moves to global partition and extract graph
      START_TIMER("Apply moves");
      for (const auto &[node, group, weight, gain, from, to] : global_move_buffer) {
        // Apply move to distributed graph
        if (is_valid_id(node)) {
          if (_p_graph.contains_global_node(node)) {
            const NodeID lnode = _p_graph.global_to_local_node(node);
            KASSERT(_p_graph.block(lnode) == from, V(lnode) << V(from));
            _p_graph.set_block(lnode, to);
          } else {
            _p_graph.set_block_weight(from, _p_graph.block_weight(from) - weight);
            _p_graph.set_block_weight(to, _p_graph.block_weight(to) + weight);
          }
        }

        // Apply move to local graph (if contained in local graph)
        if (_fm_ctx.revert_local_moves_after_batch && is_valid_id(node)) {
          if (auto it = reverse_node_mapping.find(node + 1); it != reverse_node_mapping.end()) {
            const NodeID b_node = (*it).second;
            bp_graph->set_block(b_node, to);
            shared.gain_cache.move(*bp_graph, b_node, from, to);
          }
        } else if (!_fm_ctx.revert_local_moves_after_batch && is_invalid_id(node)) {
          if (auto it = reverse_node_mapping.find(extract_id(node) + 1);
              it != reverse_node_mapping.end()) {
            const NodeID b_node = (*it).second;
            bp_graph->set_block(b_node, to);
            shared.gain_cache.move(*bp_graph, b_node, from, to);
          }
        }
      }

      // Block weightgs in the batch graph are no longer valid, so we must copy them from the
      // global graph again
      for (const BlockID block : _p_graph.blocks()) {
        bp_graph->set_block_weight(block, _p_graph.block_weight(block));
      }

      KASSERT(
          graph::debug::validate_partition(_p_graph),
          "global partition in inconsistent state after round " << global_round << "/"
                                                                << local_round,
          assert::heavy
      );
    }

    if (_fm_ctx.rebalance_after_each_global_iteration) {
      TIMED_SCOPE("Rebalance") {
        balancer->refine();
      };
    }
  }

  if (!_fm_ctx.rebalance_after_each_global_iteration && _fm_ctx.rebalance_after_refinement) {
    TIMED_SCOPE("Rebalance") {
      balancer->refine();
    };
  }

  return false;
}
} // namespace kaminpar::dist
