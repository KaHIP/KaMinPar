/*******************************************************************************
 * Coarsener that is optimized to contract clusterings.
 *
 * @file:   threshold_sparsifying_cluster_coarsener.cc
 * @author: Dominik Rosch, Daniel Seemaier
 * @date:   28.03.2025
 ******************************************************************************/
#include "kaminpar-shm/coarsening/threshold_sparsifying_cluster_coarsener.h"

#include <stdexcept>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"
#include "kaminpar-shm/coarsening/sparsification/contraction/sparsifying_cluster_contraction.h"
#include "kaminpar-shm/coarsening/sparsification/sparsification_utils.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

}

ThresholdSparsifyingClusterCoarsener::ThresholdSparsifyingClusterCoarsener(
    const Context &ctx, const PartitionContext &p_ctx
)
    : AbstractClusterCoarsener(ctx, p_ctx),
      _s_ctx(ctx.sparsification) {}

void ThresholdSparsifyingClusterCoarsener::use_communities(std::span<const NodeID>) {
  throw std::logic_error("not implemented");
}

EdgeID ThresholdSparsifyingClusterCoarsener::sparsification_target(
    const EdgeID old_m, const NodeID old_n, const EdgeID new_n
) const {
  const double target = std::min(
      _s_ctx.reduction_target_factor * old_m, _s_ctx.density_target_factor * old_m / old_n * new_n
  );
  return target < old_m ? static_cast<EdgeID>(target) : old_m;
}

bool ThresholdSparsifyingClusterCoarsener::coarsen() {
  SCOPED_HEAP_PROFILER("Level", std::to_string(_hierarchy.size()));
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()));

  START_HEAP_PROFILER("Allocation");
  RECORD("clustering") StaticArray<NodeID> clustering(current().n(), static_array::noinit);
  STOP_HEAP_PROFILER();

  const bool free_allocated_memory = !keep_allocated_memory();
  const NodeID prev_n = current().n();

  compute_clustering_for_current_graph(clustering);

  START_HEAP_PROFILER("Contract graph");
  auto coarsened = TIMED_SCOPE("Contract graph") {
    return contract_clustering(
        current(), std::move(clustering), _c_ctx.contraction, _contraction_m_ctx
    );
  };
  KASSERT(coarsened->get().m() % 2 == 0u, "graph should be undirected", assert::always);

  const EdgeID target_sparsified_m = [&] {
    if (_hierarchy.empty()) {
      return sparsification_target(_input_graph->m(), _input_graph->n(), coarsened->get().n());
    } else {
      return sparsification_target(
          _hierarchy.back()->get().m(), _hierarchy.back()->get().n(), coarsened->get().n()
      );
    }
  }();
  const EdgeID unsparsified_m = coarsened->get().m();

  DBG << "Sparsify from " << unsparsified_m << " to " << target_sparsified_m
      << " edges, but only if the factor is >=" << _s_ctx.laziness_factor;

  if (unsparsified_m > _s_ctx.laziness_factor * target_sparsified_m) {
    using namespace sparsification;
    using namespace contraction;

    auto *coarsened_downcast = dynamic_cast<CoarseGraphImpl *>(coarsened.get());
    StaticArray<NodeID> mapping = std::move(coarsened_downcast->get_mapping());
    Graph graph = std::move(coarsened_downcast->get());
    CSRGraph &csr = graph.concretize<CSRGraph>();

    const NodeID c_n = csr.n();
    const EdgeID c_m = csr.m();

    CSRGraph sparsified = [&] {
      if (_s_ctx.recontract) {
        StaticArray<EdgeWeight> edge_weights = csr.take_raw_edge_weights();

        // Free the contracted, unsparsified graph before we determine the threshold edge weight
        // This reduces overall peak memory.
        // !! This invalidates the `csr` reference !!
        {
          ((void)std::move(graph));
        }

        const utils::K_SmallestInfo<EdgeWeight> threshold = TIMED_SCOPE("Threshold selection") {
          return utils::quickselect_k_smallest<EdgeWeight>(
              c_m - target_sparsified_m + 1, edge_weights.begin(), edge_weights.end()
          );
        };

        // Free the old edge weights before we re-contract the graph to reduce peak memory.
        edge_weights.free();

        const EdgeWeight threshold_weight = threshold.value;
        const EdgeID number_of_elements_larger =
            c_m - threshold.number_of_elements_equal - threshold.number_of_elements_smaller;
        KASSERT(
            number_of_elements_larger <= target_sparsified_m, "quickselect failed", assert::always
        );

        const EdgeID number_of_equal_elements_to_include =
            target_sparsified_m - number_of_elements_larger;
        const double threshold_probability =
            1.0 * number_of_equal_elements_to_include / threshold.number_of_elements_equal;

        DBG << "Threshold weight: " << threshold_weight;
        DBG << "Threshold probability: " << threshold_probability;

        auto recoarsened = TIMED_SCOPE("Recontract and sparsify") {
          // Contraction internals: we can re-use the mapping[] array from the previous contraction
          // step, but have to re-sort vertices accordingly
          fill_cluster_buckets(
              c_n, current(), mapping, _contraction_m_ctx.buckets_index, _contraction_m_ctx.buckets
          );

          return contract_and_sparsify_clustering(
              current().concretize<CSRGraph>(),
              std::move(mapping),
              c_n,
              threshold_weight,
              threshold_probability,
              _c_ctx.contraction,
              _contraction_m_ctx
          );
        };

        auto *recoarsened_downcast = dynamic_cast<CoarseGraphImpl *>(recoarsened.get());

        // Contraction code is racy: mapping might change
        mapping = std::move(recoarsened_downcast->get_mapping());
        return std::move(recoarsened->get().concretize<CSRGraph>());
      } else {
        return remove_negative_edges(
            sparsify_and_make_negative_edges(std::move(csr), target_sparsified_m)
        );
      }
    }();

    const EdgeID sparsified_m = sparsified.m();

    _hierarchy.push_back(
        std::make_unique<contraction::CoarseGraphImpl>(
            Graph(std::make_unique<CSRGraph>(std::move(sparsified))), std::move(mapping)
        )
    );

    LOG << "Sparsified from " << unsparsified_m << " to " << sparsified_m
        << " edges (target: " << target_sparsified_m << ")";
    LOG;
  } else {
    DBG << "Coarse graph does not require sparsification";
    _hierarchy.push_back(std::move(coarsened));
  }
  STOP_HEAP_PROFILER();

  const NodeID next_n = current().n();
  const bool converged = (1.0 - 1.0 * next_n / prev_n) <= _c_ctx.convergence_threshold;

  if (free_allocated_memory) {
    _contraction_m_ctx.buckets.free();
    _contraction_m_ctx.buckets_index.free();
    _contraction_m_ctx.all_buffered_nodes.free();
  }

  return !converged;
}

CSRGraph ThresholdSparsifyingClusterCoarsener::sparsify_and_make_negative_edges(
    CSRGraph csr, const NodeID target_m
) const {
  using namespace sparsification;

  const utils::K_SmallestInfo<EdgeWeight> threshold = TIMED_SCOPE("Threshold selection") {
    return utils::quickselect_k_smallest<EdgeWeight>(
        csr.m() - target_m + 1, csr.raw_edge_weights().begin(), csr.raw_edge_weights().end()
    );
  };

  TIMED_SCOPE("Edge selection") {
    const EdgeID number_of_elements_larger =
        csr.m() - threshold.number_of_elements_equal - threshold.number_of_elements_smaller;
    KASSERT(number_of_elements_larger <= target_m, "quickselect failed", assert::always);

    const EdgeID number_of_equal_elements_to_include = target_m - number_of_elements_larger;
    const double inclusion_probability_if_equal =
        1.0 * number_of_equal_elements_to_include / threshold.number_of_elements_equal;

    DBG << "Threshold weight: " << threshold.value;
    DBG << "Threshold probability: " << inclusion_probability_if_equal;

    utils::parallel_for_upward_edges(csr, [&](const EdgeID e) {
      if (csr.edge_weight(e) > threshold.value ||
          (csr.edge_weight(e) == threshold.value &&
           Random::instance().random_bool(inclusion_probability_if_equal))) {
        csr.raw_edge_weights()[e] *= -1;
      }
    });
  };

  return csr;
}

CSRGraph ThresholdSparsifyingClusterCoarsener::remove_negative_edges(CSRGraph g) const {
  SCOPED_TIMER("Build Sparsifier");
  auto nodes = StaticArray<EdgeID>(g.n() + 1);
  sparsification::utils::parallel_for_edges_with_endpoints(g, [&](EdgeID e, NodeID u, NodeID v) {
    if (g.edge_weight(e) < 0) {
      __atomic_add_fetch(&nodes[u + 1], 1, __ATOMIC_RELAXED);
      __atomic_add_fetch(&nodes[v + 1], 1, __ATOMIC_RELAXED);
    }
  });
  parallel::prefix_sum(nodes.begin(), nodes.end(), nodes.begin());

  auto edges_added = StaticArray<EdgeID>(g.n());
  auto edges = StaticArray<NodeID>(nodes[g.n()]);
  auto edge_weights = StaticArray<EdgeWeight>(nodes[g.n()]);

  sparsification::utils::parallel_for_edges_with_endpoints(g, [&](EdgeID e, NodeID u, NodeID v) {
    if (g.edge_weight(e) < 0) {
      auto v_edges_added = __atomic_fetch_add(&edges_added[v], 1, __ATOMIC_RELAXED);
      auto u_edges_added = __atomic_fetch_add(&edges_added[u], 1, __ATOMIC_RELAXED);
      edges[nodes[v] + v_edges_added] = u;
      edges[nodes[u] + u_edges_added] = v;
      edge_weights[nodes[v] + v_edges_added] = -g.edge_weight(e);
      edge_weights[nodes[u] + u_edges_added] = -g.edge_weight(e);
    }
  });

  return CSRGraph(
      std::move(nodes),
      std::move(edges),
      g.take_raw_node_weights(),
      std::move(edge_weights),
      g.sorted()
  );
}

} // namespace kaminpar::shm
