/*******************************************************************************
 * Parallel k-way label propagation refiner.
 *
 * @file:   lp_refiner.cc
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#include "kaminpar-shm/refinement/lp/lp_refiner.h"

#include "kaminpar-shm/label_propagation/active_set.h"
#include "kaminpar-shm/label_propagation/chunk_random_iteration.h"
#include "kaminpar-shm/label_propagation/config.h"
#include "kaminpar-shm/label_propagation/node_processor.h"
#include "kaminpar-shm/label_propagation/overload_aware_selection.h"

#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

//
// Configuration
//

struct LPRefinerConfig : public LabelPropagationConfig {
  using ClusterID = BlockID;
  using ClusterWeight = BlockWeight;
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, NodeID, rm_backyard::SparseMap>;
  static constexpr bool kUseHardWeightConstraint = true;
  static constexpr bool kReportEmptyClusters = false;
};

//
// ClusterOps: the plain struct that satisfies the ClusterOps concept.
// Replaces all CRTP hooks from the old LPRefinerImpl.
//

template <typename Graph> struct LPRefinerOps {
  PartitionedGraph *p_graph = nullptr;
  const PartitionContext *p_ctx = nullptr;
  std::span<const NodeID> communities;

  BlockID cluster(const NodeID u) {
    return p_graph->block(u);
  }

  void move_node(const NodeID u, const BlockID block) {
    p_graph->set_block<false>(u, block);
  }

  BlockWeight cluster_weight(const BlockID b) {
    return p_graph->block_weight(b);
  }

  bool move_cluster_weight(
      const BlockID old_block,
      const BlockID new_block,
      const BlockWeight delta,
      const BlockWeight max_weight
  ) {
    return p_graph->move_block_weight(
        old_block, new_block, delta, max_weight, min_cluster_weight(old_block)
    );
  }

  BlockWeight max_cluster_weight(const BlockID block) {
    return p_ctx->max_block_weight(block);
  }

  BlockWeight min_cluster_weight(const BlockID block) {
    return p_ctx->min_block_weight(block);
  }

  BlockID initial_cluster(const NodeID u) {
    return p_graph->block(u);
  }

  BlockWeight initial_cluster_weight(const BlockID b) {
    return p_graph->block_weight(b);
  }

  bool accept_neighbor(const NodeID u, const NodeID v) {
    return communities.empty() || communities[u] == communities[v];
  }

  // Not used by overload-aware selection, but needed by the processor for completeness.
  bool accept_cluster(const BlockID /* current_cluster */, const BlockID /* initial_cluster */) {
    return true;
  }

  void init_cluster(const NodeID /* u */, const BlockID /* b */) {}

  void init_cluster_weight(const BlockID /* b */, const BlockWeight /* weight */) {}

  void reassign_cluster_weights(
      const StaticArray<BlockID> & /* mapping */, const BlockID /* num_new_clusters */
  ) {}

  bool skip_node(const NodeID /* u */) {
    return false;
  }

  void reset_node_state(const NodeID /* u */) {}

  bool activate_neighbor(const NodeID /* v */) {
    return true;
  }
};

//
// Actual implementation -- composition, no CRTP
//

template <typename Graph> class LPRefinerImpl {
  SET_DEBUG(true);

  using Config = LPRefinerConfig;
  using ClusterID = Config::ClusterID;
  using Ops = LPRefinerOps<Graph>;
  using Selection = lp::OverloadAwareClusterSelection<Ops>;
  using Processor = lp::LPNodeProcessor<Graph, Ops, Selection, Config>;
  using Iterator = lp::ChunkRandomIterator<Config>;

  static constexpr bool kUseActiveSet =
      Config::kUseActiveSetStrategy || Config::kUseLocalActiveSetStrategy;
  static constexpr std::size_t kInfiniteIterations = std::numeric_limits<std::size_t>::max();

public:
  using Permutations = Iterator::Permutations;

  // Data structures for memory reuse between calls.
  using DataStructures = std::tuple<typename Processor::DataStructures, typename Iterator::DataStructures>;

  LPRefinerImpl(const Context &ctx, Permutations &permutations)
      : _r_ctx(ctx.refinement),
        _ops(),
        _selection(_ops, _r_ctx.lp.tie_breaking_strategy),
        _processor(_ops, _selection, _active_set),
        _iterator(permutations),
        _max_degree(_r_ctx.lp.large_degree_threshold),
        _impl(_r_ctx.lp.impl) {
    _processor.set_max_num_neighbors(_r_ctx.lp.max_num_neighbors);
    _num_nodes = ctx.partition.n;
    _num_clusters = ctx.partition.k;
  }

  void allocate() {
    SCOPED_HEAP_PROFILER("Allocation");
    SCOPED_TIMER("Allocation");

    _processor.allocate(_num_nodes, _num_nodes, _num_clusters);
  }

  void initialize(const Graph *graph) {
    _graph = graph;
  }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    KASSERT(_graph == p_graph.graph().underlying_graph());
    KASSERT(p_graph.k() <= p_ctx.k);
    SCOPED_HEAP_PROFILER("Label Propagation");

    _ops.p_graph = &p_graph;
    _ops.p_ctx = &p_ctx;

    _processor.initialize(_graph, p_ctx.k);
    _iterator.clear();

    const std::size_t max_iterations =
        _r_ctx.lp.num_iterations == 0 ? kInfiniteIterations : _r_ctx.lp.num_iterations;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration));
      if (perform_iteration() == 0) {
        break;
      }
    }

    return true;
  }

  void set_communities(std::span<const NodeID> communities) {
    _ops.communities = communities;
  }

  void setup(DataStructures structs) {
    auto [proc_structs, iter_structs] = std::move(structs);
    _processor.setup(std::move(proc_structs));
    _iterator.setup(std::move(iter_structs));
  }

  DataStructures release() {
    return std::make_tuple(_processor.release(), _iterator.release());
  }

private:
  NodeID perform_iteration() {
    if (_iterator.empty()) {
      _iterator.init_chunks(*_graph, 0, _graph->n(), _max_degree);
    }
    _iterator.shuffle_chunks();

    auto &current_num_clusters = _processor.current_num_clusters();

    auto handler = [&](const NodeID u) {
      auto &rand = Random::instance();
      auto &rating_map = _processor.rating_map_ets().local();
      auto &tb = _processor.tie_breaking_clusters_ets().local();
      auto &tbf = _processor.tie_breaking_favored_clusters_ets().local();
      return _processor.handle_node(u, rand, rating_map, tb, tbf);
    };

    auto first_phase_handler = [&](const NodeID u) {
      auto &rand = Random::instance();
      auto &rating_map = _processor.rating_map_ets().local();
      auto &tb = _processor.tie_breaking_clusters_ets().local();
      auto &tbf = _processor.tie_breaking_favored_clusters_ets().local();
      return _processor.handle_first_phase_node(u, rand, rating_map, tb, tbf);
    };

    auto should_stop = [&] { return false; }; // refiner doesn't use early stopping
    auto is_active = [&](const NodeID u) { return _processor.is_active(u); };

    switch (_impl) {
    case LabelPropagationImplementation::GROWING_HASH_TABLES:
    case LabelPropagationImplementation::SINGLE_PHASE:
      return _iterator.iterate(*_graph, _max_degree, handler, should_stop, is_active, current_num_clusters);
    case LabelPropagationImplementation::TWO_PHASE: {
      const auto [num_processed, num_moved_first] = _iterator.iterate_first_phase(
          *_graph, _max_degree, first_phase_handler, should_stop, is_active, current_num_clusters
      );

      auto &second_phase_nodes = _processor.second_phase_nodes();
      NodeID total_moved = num_moved_first;

      if (!second_phase_nodes.empty()) {
        const std::size_t num_clusters = _processor.initial_num_clusters();
        auto &concurrent_map = _processor.concurrent_rating_map();
        if (concurrent_map.capacity() < num_clusters) {
          concurrent_map.resize(num_clusters);
        }

        auto &rand = Random::instance();
        for (const NodeID u : second_phase_nodes) {
          const auto [moved_node, emptied_cluster] =
              _processor.handle_second_phase_node(u, rand, concurrent_map);

          if (moved_node) {
            ++total_moved;
          }
          if (emptied_cluster) {
            --current_num_clusters;
          }
        }

        second_phase_nodes.clear();
      }

      return total_moved;
    }
    }

    __builtin_unreachable();
  }

  // --- Members ---

  const RefinementContext &_r_ctx;

  // Building blocks (composition)
  ActiveSet<kUseActiveSet> _active_set;
  Ops _ops;
  Selection _selection;
  Processor _processor;
  Iterator _iterator;

  const Graph *_graph = nullptr;
  NodeID _num_nodes = 0;
  BlockID _num_clusters = 0;
  NodeID _max_degree;
  LabelPropagationImplementation _impl;
};

//
// Wrapper that handles CSR vs Compressed graph dispatch + memory reuse
//

class LPRefinerImplWrapper {
public:
  LPRefinerImplWrapper(const Context &ctx)
      : _csr_impl(std::make_unique<LPRefinerImpl<CSRGraph>>(ctx, _permutations)),
        _compressed_impl(std::make_unique<LPRefinerImpl<CompressedGraph>>(ctx, _permutations)) {}

  void initialize(const PartitionedGraph &p_graph) {
    reified(
        p_graph,
        [&](const auto &graph) { _csr_impl->initialize(&graph); },
        [&](const auto &graph) { _compressed_impl->initialize(&graph); }
    );
  }

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
    SCOPED_TIMER("Label Propagation");

    const auto refine = [&](auto &impl) {
      if (_freed) {
        _freed = false;
        impl.allocate();
      } else {
        impl.setup(std::move(_structs));
      }

      const bool found_improvement = impl.refine(p_graph, p_ctx);

      _structs = impl.release();
      return found_improvement;
    };

    return reified(
        p_graph,
        [&](const auto &) { return refine(*_csr_impl); },
        [&](const auto &) { return refine(*_compressed_impl); }
    );
  }

  void set_communities(std::span<const NodeID> communities) {
    _csr_impl->set_communities(communities);
    _compressed_impl->set_communities(communities);
  }

private:
  std::unique_ptr<LPRefinerImpl<CSRGraph>> _csr_impl;
  std::unique_ptr<LPRefinerImpl<CompressedGraph>> _compressed_impl;

  // The data structures which are used by the LP refiner and are shared between the
  // different implementations.
  bool _freed = true;
  LPRefinerImpl<Graph>::Permutations _permutations;
  LPRefinerImpl<Graph>::DataStructures _structs;
};

//
// Exposed wrapper
//

LabelPropagationRefiner::LabelPropagationRefiner(const Context &ctx)
    : _impl_wrapper(std::make_unique<LPRefinerImplWrapper>(ctx)) {}

LabelPropagationRefiner::~LabelPropagationRefiner() = default;

std::string LabelPropagationRefiner::name() const {
  return "Label Propagation";
}

void LabelPropagationRefiner::initialize(const PartitionedGraph &p_graph) {
  _impl_wrapper->initialize(p_graph);
}

bool LabelPropagationRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return _impl_wrapper->refine(p_graph, p_ctx);
}

void LabelPropagationRefiner::set_communities(std::span<const NodeID> communities) {
  _impl_wrapper->set_communities(communities);
}

} // namespace kaminpar::shm
