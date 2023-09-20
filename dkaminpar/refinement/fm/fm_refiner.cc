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
#include <tbb/parallel_invoke.h>

#include "dkaminpar/algorithms/border_nodes.h"
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
#include "kaminpar/refinement/fm/fm_refiner.h"
#include "kaminpar/refinement/fm/stopping_policies.h"

#include "common/datastructures/binary_heap.h"
#include "common/datastructures/rating_map.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
namespace {
SET_STATISTICS_FROM_GLOBAL();
SET_DEBUG(true);
} // namespace

namespace fm {
NodeMapper::NodeMapper(NoinitVector<GlobalNodeID> batch_to_graph)
    : _batch_to_graph(std::move(batch_to_graph)) {
  construct();
}

inline GlobalNodeID NodeMapper::to_graph(NodeID bnode) const {
  KASSERT(bnode < _batch_to_graph.size());
  return _batch_to_graph[bnode];
}

inline NodeID NodeMapper::to_batch(GlobalNodeID gnode) const {
  auto it = _graph_to_batch.find(gnode + 1);
  return it != _graph_to_batch.end() ? (*it).second : kInvalidNodeID;
}

void NodeMapper::construct() {
  tbb::parallel_for<std::size_t>(0, _batch_to_graph.size(), [&](const std::size_t i) {
    KASSERT(
        _graph_to_batch.find(_batch_to_graph[i] + 1) == _graph_to_batch.end(),
        "both nodes " << i << " and " << (*_graph_to_batch.find(_batch_to_graph[i] + 1)).second
                      << " from the batch graph are mapped to the global node "
                      << _batch_to_graph[i] << " == "
                      << _batch_to_graph[(*_graph_to_batch.find(_batch_to_graph[i] + 1)).second]
    );
    _graph_to_batch.insert(_batch_to_graph[i] + 1, i);
  });
}

EnabledPartitionRollbacker::EnabledPartitionRollbacker(
    DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _p_graph(p_graph),
      _p_ctx(p_ctx),
      _best_cut(metrics::edge_cut(_p_graph)),
      _best_partition(_p_graph.total_n()),
      _best_block_weights(_p_graph.k()) {
  copy_partition();
}

void EnabledPartitionRollbacker::rollback() {
  if (!_last_is_best) {
    tbb::parallel_invoke(
        [&] {
          _p_graph.pfor_all_nodes([&](const NodeID u) {
            _p_graph.set_block<false>(u, _best_partition[u]);
          });
        },
        [&] {
          _p_graph.pfor_blocks([&](const BlockID b) {
            _p_graph.set_block_weight(b, _best_block_weights[b]);
          });
        }
    );
  }
}

void EnabledPartitionRollbacker::update() {
  const EdgeWeight current_cut = metrics::edge_cut(_p_graph);
  const double current_l1 = metrics::imbalance_l1(_p_graph, _p_ctx);

  // Accept if the previous best partition is imbalanced and we improved its balance
  // OR if we are balanced and got a better cut than before
  if ((_best_l1 > 0 && current_l1 < _best_l1) || (current_l1 == 0 && current_cut <= _best_cut)) {
    copy_partition();
    _best_cut = current_cut;
    _best_l1 = current_l1;
    _last_is_best = true;
  } else {
    _last_is_best = false;
  }
}

void EnabledPartitionRollbacker::copy_partition() {
  tbb::parallel_invoke(
      [&] {
        _p_graph.pfor_all_nodes([&](const NodeID u) { _best_partition[u] = _p_graph.block(u); });
      },
      [&] {
        _p_graph.pfor_blocks([&](const BlockID b) {
          _best_block_weights[b] = _p_graph.block_weight(b);
        });
      }
  );
}
} // namespace fm

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

  // Track the best partition that we see during FM refinement
  std::unique_ptr<fm::PartitionRollbacker> rollbacker =
      [&]() -> std::unique_ptr<fm::PartitionRollbacker> {
    if (_fm_ctx.rollback_deterioration) {
      return std::make_unique<fm::EnabledPartitionRollbacker>(_p_graph, _p_ctx);
    } else {
      return std::make_unique<fm::DisabledPartitionRollbacker>();
    }
  }();

  for (std::size_t global_round = 0; global_round < _fm_ctx.num_global_iterations; ++global_round) {
    const EdgeWeight initial_cut =
        _ctx.refinement.fm.use_abortion_threshold ? metrics::edge_cut(_p_graph) : -1;

    const auto seed_nodes = _fm_ctx.use_independent_seeds
                                ? graph::find_independent_border_set(_p_graph, global_round)
                                : graph::find_border_nodes(_p_graph);

    // Run distributed BFS to extract batches around the seed nodes
    graph::BfsExtractor bfs_extractor(_p_graph.graph());
    bfs_extractor.initialize(_p_graph);
    bfs_extractor.set_max_hops(_fm_ctx.max_hops);
    bfs_extractor.set_max_radius(_fm_ctx.max_radius);
    auto extraction_result = bfs_extractor.extract(seed_nodes);

    // This is the batch graph containing the subgraphs around all seed nodes
    // The last `k` nodes are pseudo-nodes representing the blocks of the graph partition
    shm::Graph *b_graph = extraction_result.graph.get();
    shm::PartitionedGraph *bp_graph = extraction_result.p_graph.get();

    DBG << "BFS extraction result: n=" << b_graph->n() << ", m=" << b_graph->m();
    KASSERT(
        shm::validate_graph(*b_graph, false, _p_graph.k()),
        "BFS extractor returned invalid graph data structure",
        assert::normal
    );

    // Overwrite the block weights in the batch graph with block weights of the global graph
    for (const BlockID block : _p_graph.blocks()) {
      bp_graph->set_block_weight(block, _p_graph.block_weight(block));
    }

    // @todo obvious optimization: avoid re-allocation
    START_TIMER("Prepare thread-local workers");
    const shm::PartitionContext shm_p_ctx = setup_shm_p_ctx(*b_graph);
    const fm::NodeMapper node_mapper(extraction_result.node_mapping);
    const shm::KwayFMRefinementContext shm_fm_ctx = setup_fm_ctx();
    shm::fm::SharedData shared = setup_fm_data(*bp_graph, seed_nodes, node_mapper);

    // Create thread-local workers numbered 1..P
    std::atomic<int> next_id = 0;
    tbb::enumerable_thread_specific<shm::LocalizedFMRefiner> worker_ets([&] {
      // It is important that worker IDs start at 1, otherwise the node
      // tracker won't work since 0 is reserved for "unmarked / untracked"
      shm::LocalizedFMRefiner worker(++next_id, shm_p_ctx, shm_fm_ctx, *bp_graph, shared);

      // This allows access to the moves that were applied to the shared bp_graph partition
      worker.enable_move_recording();

      return worker;
    });
    STOP_TIMER();

    std::vector<GlobalMove> move_sets;

    for (std::size_t local_round = 0; local_round < _fm_ctx.num_local_iterations; ++local_round) {
      DBG0 << "Starting FM round " << global_round << "/" << local_round;

      // Unlock previously locked nodes, etc ...
      // @todo re-init border nodes if not using the independent set as FM seeds
      prepare_shared_data_for_local_round(*bp_graph, shared);

      // Compute the number of seeds per chunk / batch
      const NodeID total_num_seeds = shared.border_nodes.size();
      const NodeID total_num_chunks =
          _fm_ctx.chunk_local_rounds
              ? std::min<NodeID>(_fm_ctx.chunks.compute(_ctx.parallel), total_num_seeds)
              : 1;
      const NodeID num_seeds_per_chunk = std::ceil(1.0 * total_num_seeds / total_num_chunks);
      bool have_more_seeds = true;

      DBG0 << "Number of seeds on PE 0: " << total_num_seeds;
      DBG0 << "Number of chunks on PE 0: " << total_num_chunks;
      DBG0 << "Seeds per chunk on PE 0: " << num_seeds_per_chunk;

      // Perform local search on the current chunk / batch, synchronize and apply moves to the
      // distributed graph ...
      do {
        move_sets.clear();

        START_TIMER("Thread-local FM");
        for (NodeID progress = 0; shared.border_nodes.has_more() && progress < num_seeds_per_chunk;
             ++progress) {
          // Perform the FM search starting at the next seed
          shm::LocalizedFMRefiner &worker = worker_ets.local();
          const EdgeWeight gain = worker.run_batch();
          const NodeID seed = worker.last_batch_seed_nodes().front();
          const auto &moves = worker.last_batch_moves();

          // Zero seeds are possible if we do not unlock moved nodes after a batch
          KASSERT(worker.last_batch_seed_nodes().size() <= 1, "expected exactly one seed node");
          KASSERT(
              moves.empty() || !worker.last_batch_seed_nodes().empty(),
              "search did not get any seeds nodes, yet performed some moves"
          );

          if (!moves.empty()) {
            // Unique identifier for this set of moves == global node ID of the FM seed
            // const GlobalNodeID group = node_mapper.to_graph(seed);
            //
            // Since the global conflict resolver uses the upper 32 bits to store the owning PE, we
            // can use the local seed ID and still have groups that are unique globally
            const GlobalNodeID group = seed;

            bool print_delim = false;
            for (const auto &[node, from, improvement] : moves) {
              if (node_mapper.to_graph(node) == 35857) {
                DBG << "Recording applied move with seed " << seed << ": "
                    << node_mapper.to_graph(node) << " /// " << node << " /// group " << group
                    << " ::: " << from << " --> " << bp_graph->block(node);
                print_delim = true;
              }

              move_sets.push_back(GlobalMove{
                  .node = node_mapper.to_graph(node),
                  .group = group,
                  .weight = static_cast<NodeWeight>(bp_graph->node_weight(node)),
                  .gain = gain,
                  .from = from,
                  .to = bp_graph->block(node),
              });
            }
            if (print_delim)
              DBG << "-----";

            // Revert changes to the batch graph to make local FM searches independent of each other
            if (_fm_ctx.revert_local_moves_after_batch) {
              for (const auto &[node, from, improvement] : moves) {
                const BlockID to = bp_graph->block(node);
                bp_graph->set_block(node, from);
                shared.gain_cache.move(*bp_graph, node, to, from);

                if (node != seed) {
                  shared.node_tracker.set(node, shm::fm::NodeTracker::UNLOCKED);
                }
              }
            }
          }
        }
        STOP_TIMER();

        mpi::barrier(_p_graph.communicator());

        // Resolve global move conflicts: after this operation, global_move_buffer accepted moves
        // with valid node IDs (is_valid_id(node)) and rejected moves with invalid node IDs
        // (is_invalid_id(node))
        START_TIMER("Resolve move conflicts");
        auto global_move_buffer =
            broadcast_and_resolve_global_moves(move_sets, _p_graph.communicator());
        STOP_TIMER();

        mpi::barrier(_p_graph.communicator());

        // Apply moves to global partition and extract graph
        START_TIMER("Apply moves");
        for (const auto &[node, group, weight, gain, from, to] : global_move_buffer) {
          // Apply move to distributed graph
          if (is_valid_id(node)) {
            if (_p_graph.contains_global_node(node)) {
              const NodeID lnode = _p_graph.global_to_local_node(node);
              if (lnode == 35857) {
                DBG << "expected " << lnode << " to be in block " << from << ", move to " << to
                    << " /// node ID " << node << " /// group ID " << group;
              }
              KASSERT(_p_graph.block(lnode) == from, V(lnode) << V(from));
              _p_graph.set_block(lnode, to);
            } else {
              _p_graph.set_block_weight(from, _p_graph.block_weight(from) - weight);
              _p_graph.set_block_weight(to, _p_graph.block_weight(to) + weight);
            }
          }

          // Apply move to local graph (if contained in local graph)
          if (_fm_ctx.revert_local_moves_after_batch && is_valid_id(node)) {
            if (const NodeID bnode = node_mapper.to_batch(node); bnode != kInvalidNodeID) {
              bp_graph->set_block(bnode, to);
              shared.gain_cache.move(*bp_graph, bnode, from, to);
            }
          } else if (!_fm_ctx.revert_local_moves_after_batch && is_invalid_id(node)) {
            if (const NodeID bnode = node_mapper.to_batch(extract_id(node));
                bnode != kInvalidNodeID) {
              bp_graph->set_block(bnode, from);
              shared.gain_cache.move(*bp_graph, bnode, to, from);
            }
          }
        }

        // Block weights in the batch graph are no longer valid (@todo why?), so we must copy them
        // from the global graph again
        for (const BlockID block : _p_graph.blocks()) {
          bp_graph->set_block_weight(block, _p_graph.block_weight(block));
        }
        STOP_TIMER();

        KASSERT(
            graph::debug::validate_partition(_p_graph),
            "global partition in inconsistent state after round " << global_round << "/"
                                                                  << local_round,
            assert::heavy
        );

        // Continue for as long as some PE has more seeds left
        have_more_seeds = shared.border_nodes.has_more();
        MPI_Allreduce(
            MPI_IN_PLACE, &have_more_seeds, 1, MPI_C_BOOL, MPI_LOR, _p_graph.communicator()
        );
      } while (have_more_seeds);
    }

    if (_fm_ctx.rebalance_after_each_global_iteration) {
      // Since we have changed the partition, re-initialize the balancer
      balancer->initialize();
      balancer->refine();
    }
    rollbacker->update();

    if (_ctx.refinement.fm.use_abortion_threshold) {
      const EdgeWeight final_cut = metrics::edge_cut(_p_graph);
      const double improvement = 1.0 * (initial_cut - final_cut) / initial_cut;
      if (1.0 - improvement > _ctx.refinement.fm.abortion_threshold) {
        break;
      }
    }
  }

  if (!_fm_ctx.rebalance_after_each_global_iteration && _fm_ctx.rebalance_after_refinement) {
    // Since we have changed the partition, re-initialize the balancer
    balancer->initialize();
    balancer->refine();

    rollbacker->update();
  }
  rollbacker->rollback();

  return false;
}

shm::PartitionContext FMRefiner::setup_shm_p_ctx(const shm::Graph &b_graph) const {
  shm::PartitionContext shm_p_ctx;
  shm_p_ctx.epsilon = _p_ctx.epsilon;
  shm_p_ctx.k = _p_ctx.k;
  shm_p_ctx.n = asserting_cast<NodeID>(b_graph.n());
  shm_p_ctx.m = asserting_cast<EdgeID>(b_graph.m());
  shm_p_ctx.total_node_weight = asserting_cast<NodeWeight>(_p_ctx.graph->global_total_node_weight);
  shm_p_ctx.total_edge_weight = asserting_cast<EdgeWeight>(_p_ctx.graph->global_total_edge_weight);
  shm_p_ctx.max_node_weight = asserting_cast<NodeWeight>(_p_ctx.graph->global_max_node_weight);
  shm_p_ctx.setup_block_weights();
  return shm_p_ctx;
}

shm::fm::SharedData FMRefiner::setup_fm_data(
    const shm::PartitionedGraph &bp_graph,
    const std::vector<NodeID> &lseeds,
    const fm::NodeMapper &mapper
) const {
  shm::fm::SharedData shared(bp_graph.n(), bp_graph.k());

  shared.gain_cache.initialize(bp_graph);
  if (_fm_ctx.use_bfs_seeds_as_fm_seeds) {
    tbb::concurrent_vector<NodeID> bseeds;
    tbb::parallel_for<std::size_t>(0, lseeds.size(), [&](const std::size_t i) {
      const NodeID lseed = lseeds[i];
      const GlobalNodeID gseed = _p_graph.local_to_global_node(lseed);
      bseeds.push_back(mapper.to_batch(gseed));
    });
    KASSERT(std::find(bseeds.begin(), bseeds.end(), kInvalidNodeID) == bseeds.end());
    shared.border_nodes.init_precomputed(bp_graph, bseeds);
  } else {
    shared.border_nodes.init(bp_graph);
  }
  shared.border_nodes.shuffle();

  return shared;
}

void FMRefiner::prepare_shared_data_for_local_round(
    shm::PartitionedGraph &bp_graph, shm::fm::SharedData &shared
) {
  shared.node_tracker.reset();
  for (const BlockID block : _p_graph.blocks()) {
    shared.node_tracker.set(bp_graph.n() - block - 1, shm::fm::NodeTracker::MOVED_GLOBALLY);
  }
}

shm::KwayFMRefinementContext FMRefiner::setup_fm_ctx() const {
  return shm::KwayFMRefinementContext{
      .num_seed_nodes = 1,
      .alpha = _fm_ctx.alpha,
      .num_iterations = 1,
      .unlock_seed_nodes = false,
      .use_exact_abortion_threshold = false,
      .abortion_threshold = 0.999,
  };
}
} // namespace kaminpar::dist
