/*******************************************************************************
 * Meta-refiner that refines an ensemble-contracted graph and the fine graph.
 *
 * @file:   meta_refiner.cc
 ******************************************************************************/
#include "kaminpar-shm/refinement/meta_refiner.h"

#include <stdexcept>
#include <string>

#include <tbb/parallel_for.h>

#include "kaminpar-shm/coarsening/clustering/ensemble_clusterer.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

MetaRefiner::MetaRefiner(
    const Context &ctx, std::unique_ptr<Clusterer> lp_clusterer, std::unique_ptr<Refiner> refiner
)
    : _ctx(ctx),
      _ensemble_clusterer(std::move(lp_clusterer), ctx.refinement.meta.num_clusterings),
      _refiner(std::move(refiner)) {
  if (_refiner == nullptr) {
    throw std::invalid_argument("meta refiner requires an underlying refiner");
  }
}

std::string MetaRefiner::name() const {
  return "Meta Refiner (" + _refiner->name() + ")";
}

void MetaRefiner::initialize([[maybe_unused]] const PartitionedGraph &p_graph) {}

bool MetaRefiner::refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  SCOPED_HEAP_PROFILER("Meta Refinement");
  SCOPED_TIMER("Meta Refinement");

  _refiner->set_output_level(_output_level);
  _refiner->set_output_prefix(_output_prefix);

  if (p_graph.n() == 0) {
    _refiner->set_communities(_communities);
    return run_refiner(p_graph, p_ctx);
  }

  StaticArray<NodeID> clustering(p_graph.n(), static_array::noinit);
  configure_clusterer(_ensemble_clusterer, p_graph.graph(), _ctx, p_ctx);
  _ensemble_clusterer.set_communities(_communities);
  _ensemble_clusterer.compute_clustering(clustering, p_graph.graph(), true);

  // The LP clusterings may cross block boundaries. Intersect the ensemble with the current
  // partition so that each coarse node has a unique initial block.
  StaticArray<NodeID> partition_clustering(p_graph.n(), static_array::noinit);
  tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID u) {
    partition_clustering[u] = p_graph.block(u);
  });
  clustering = overlay_clusterings(p_graph.graph(), std::move(clustering), partition_clustering);

  // Also preserve any externally imposed communities, e.g. during restricted V-cycles.
  if (!_communities.empty()) {
    StaticArray<NodeID> communities(_communities.begin(), _communities.end());
    clustering = overlay_clusterings(p_graph.graph(), std::move(clustering), communities);
  }

  auto coarse_graph = TIMED_SCOPE("Contract graph") {
    return contract_clustering(p_graph.graph(), std::move(clustering), _ctx.coarsening.contraction);
  };

  bool coarse_improvement = false;
  if (coarse_graph->get().n() < p_graph.n()) {
    StaticArray<BlockID> coarse_partition(coarse_graph->get().n());
    coarse_graph->project_down(p_graph.raw_partition(), coarse_partition);
    PartitionedGraph coarse_p_graph(coarse_graph->get(), p_graph.k(), std::move(coarse_partition));

    StaticArray<NodeID> coarse_communities;
    if (!_communities.empty()) {
      coarse_communities.resize(coarse_p_graph.n(), static_array::noinit);
      project_communities(*coarse_graph, _communities, coarse_communities);
    }

    _refiner->set_communities(coarse_communities);
    coarse_improvement = run_refiner(coarse_p_graph, p_ctx);

    StaticArray<BlockID> fine_partition(p_graph.n());
    coarse_graph->project_up(coarse_p_graph.raw_partition(), fine_partition);
    p_graph = PartitionedGraph(p_graph.graph(), p_graph.k(), std::move(fine_partition));
  }

  _refiner->set_communities(_communities);
  const bool fine_improvement = run_refiner(p_graph, p_ctx);

  return coarse_improvement || fine_improvement;
}

void MetaRefiner::set_communities(const std::span<const NodeID> communities) {
  _communities = communities;
}

bool MetaRefiner::run_refiner(PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  _refiner->initialize(p_graph);
  return _refiner->refine(p_graph, p_ctx);
}

} // namespace kaminpar::shm
