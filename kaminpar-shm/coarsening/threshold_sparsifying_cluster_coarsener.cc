/*******************************************************************************
 * Coarsener that is optimized to contract clusterings.
 *
 * @file:   threshold_sparsifying_cluster_coarsener.cc
 * @author: Dominik Rosch, Daniel Seemaier
 * @date:   28.03.2025
 ******************************************************************************/
#include "kaminpar-shm/coarsening/threshold_sparsifying_cluster_coarsener.h"

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
  KAMINPAR_NOT_IMPLEMENTED_ERROR("communities not implemented by this coarsener");
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
        // Free the contracted, unsparsified graph before we determine the threshold edge weight
        // This reduces overall peak memory.
        // !! This invalidates the `csr` reference !!
        StaticArray<EdgeWeight> c_edge_weights = csr.take_raw_edge_weights();
        {
          ((void)std::move(graph));
        }

        // Sorted cluster buckets might have changed due to the node remapping --> re-do
        // preprocessing for cluster contraction
        fill_cluster_buckets(
            c_n, current(), mapping, _contraction_m_ctx.buckets_index, _contraction_m_ctx.buckets
        );

        auto recontracted = [&]() {
          SCOPED_TIMER("Sparsification");
          if (_s_ctx.algorithm == SparsificationAlgorithm::THRESHOLD) {
            KASSERT(
                _s_ctx.score_function == ScoreFunctionSection::WEIGHT,
                "not implemented",
                assert::always
            );

            return recontract_with_threshold_sparsification(
                c_n, c_m, std::move(c_edge_weights), std::move(mapping), target_sparsified_m
            );
          } else if (_s_ctx.algorithm == SparsificationAlgorithm::UNIFORM_RANDOM_SAMPLING) {
            return recontract_with_uniform_sampling(
                c_n, c_m, std::move(c_edge_weights), std::move(mapping), target_sparsified_m
            );
          } else if (_s_ctx.algorithm ==
                     SparsificationAlgorithm::WEIGHTED_UNIFORM_RANDOM_SAMPLING) {
            return recontract_with_weighted_uniform_sampling(
                c_n, c_m, std::move(c_edge_weights), std::move(mapping), target_sparsified_m
            );
          }

          KAMINPAR_NOT_IMPLEMENTED_ERROR("sparsification configuration not available");
        }();

        // Contraction code is racy: mapping might have changed
        auto *recontracted_impl = dynamic_cast<CoarseGraphImpl *>(recontracted.get());
        mapping = std::move(recontracted_impl->get_mapping());
        return std::move(recontracted->get().concretize<CSRGraph>());
      } else {
        if (_s_ctx.algorithm == SparsificationAlgorithm::THRESHOLD &&
            _s_ctx.score_function == ScoreFunctionSection::WEIGHT) {
          return keep_only_negative_edges(
              sparsify_and_make_negative_edges(std::move(csr), target_sparsified_m)
          );
        }

        KAMINPAR_NOT_IMPLEMENTED_ERROR("sparsification configuration not available");
      }
    }();

    const EdgeID sparsified_m = sparsified.m();

    _hierarchy.push_back(std::make_unique<contraction::CoarseGraphImpl>(
        Graph(std::make_unique<CSRGraph>(std::move(sparsified))), std::move(mapping)
    ));

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

CSRGraph ThresholdSparsifyingClusterCoarsener::keep_only_negative_edges(CSRGraph g) const {
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

std::unique_ptr<CoarseGraph>
ThresholdSparsifyingClusterCoarsener::recontract_with_threshold_sparsification(
    const NodeID c_n,
    const EdgeID c_m,
    StaticArray<EdgeWeight> c_edge_weights,
    StaticArray<NodeID> mapping,
    const EdgeID target_m
) {
  using namespace sparsification;

  if (target_m < 2) {
    return contract_and_sparsify_clustering(
        current().concretize<CSRGraph>(),
        std::move(mapping),
        c_n,
        [](const NodeID, const EdgeWeight, const NodeID) { return false; },
        _c_ctx.contraction,
        _contraction_m_ctx
    );
  }

  const utils::K_SmallestInfo<EdgeWeight> threshold = TIMED_SCOPE("Threshold selection") {
    return utils::quickselect_k_smallest<EdgeWeight>(
        c_m - target_m + 1, c_edge_weights.begin(), c_edge_weights.end()
    );
  };

  // We do not need them any longer, so free them now to reduce peak memory
  c_edge_weights.free();

  const EdgeWeight threshold_weight = threshold.value;
  const EdgeID number_of_elements_larger =
      c_m - threshold.number_of_elements_equal - threshold.number_of_elements_smaller;
  KASSERT(number_of_elements_larger <= target_m, "quickselect failed", assert::always);

  const EdgeID number_of_equal_elements_to_include = target_m - number_of_elements_larger;
  const double threshold_probability =
      1.0 * number_of_equal_elements_to_include / threshold.number_of_elements_equal;

  DBG << "Threshold weight: " << threshold_weight;
  DBG << "Threshold probability: " << threshold_probability;

  const std::size_t seed =
      Random::instance().random_index(0, std::numeric_limits<std::size_t>::max());

  auto throw_dice = [&](const NodeID u, const NodeID v) {
    std::size_t hash = ((static_cast<std::size_t>(std::max(u, v)) << 32) |
                        static_cast<std::size_t>(std::min(u, v))) +
                       seed;

    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdL;
    hash ^= hash >> 33;
    hash *= 0xc4ceb9fe1a85ec53L;
    hash ^= hash >> 33;

    hash &= (1ul << 32) - 1;
    return 1.0 * hash / ((1ul << 32) - 1) < threshold_probability;
  };

  auto sample_edge = [&](const NodeID u, const EdgeWeight w, const NodeID v) {
    return w > threshold_weight || (w == threshold_weight && throw_dice(u, v));
  };

  return contract_and_sparsify_clustering(
      current().concretize<CSRGraph>(),
      std::move(mapping),
      c_n,
      sample_edge,
      _c_ctx.contraction,
      _contraction_m_ctx
  );
}

std::unique_ptr<CoarseGraph> ThresholdSparsifyingClusterCoarsener::recontract_with_uniform_sampling(
    const NodeID c_n,
    const EdgeID c_m,
    StaticArray<EdgeWeight> c_edge_weights,
    StaticArray<NodeID> mapping,
    const EdgeID target_m
) {
  using namespace contraction;
  using namespace sparsification;

  // We do not need them any longer, so free them now to reduce peak memory
  c_edge_weights.free();

  const double probability = 1.0 * target_m / c_m;
  DBG << "Sampling probability: " << probability;

  const std::size_t seed =
      Random::instance().random_index(0, std::numeric_limits<std::size_t>::max());

  auto throw_dice = [&](const NodeID u, const NodeID v) {
    std::size_t hash = ((static_cast<std::size_t>(std::max(u, v)) << 32) |
                        static_cast<std::size_t>(std::min(u, v))) +
                       seed;

    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdL;
    hash ^= hash >> 33;
    hash *= 0xc4ceb9fe1a85ec53L;
    hash ^= hash >> 33;

    hash &= (1ul << 32) - 1;
    return 1.0 * hash / ((1ul << 32) - 1) < probability;
  };

  auto sample_edge = [&](const NodeID u, EdgeWeight /* w */, const NodeID v) {
    return throw_dice(u, v);
  };

  return contract_and_sparsify_clustering(
      current().concretize<CSRGraph>(),
      std::move(mapping),
      c_n,
      sample_edge,
      _c_ctx.contraction,
      _contraction_m_ctx
  );
}

std::unique_ptr<CoarseGraph>
ThresholdSparsifyingClusterCoarsener::recontract_with_weighted_uniform_sampling(
    const NodeID c_n,
    const EdgeID c_m,
    StaticArray<EdgeWeight> c_edge_weights,
    StaticArray<NodeID> mapping,
    const EdgeID target_m
) {
  using namespace contraction;
  using namespace sparsification;

  // We do not need them any longer, so free them now to reduce peak memory
  c_edge_weights.free();

  const double probability = 1.0 * target_m / c_m;
  DBG << "Sampling probability: " << probability;

  const std::size_t seed =
      Random::instance().random_index(0, std::numeric_limits<std::size_t>::max());

  auto throw_dice = [&](const NodeID u, const NodeID v) {
    std::size_t hash = ((static_cast<std::size_t>(std::max(u, v)) << 32) |
                        static_cast<std::size_t>(std::min(u, v))) +
                       seed;

    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdL;
    hash ^= hash >> 33;
    hash *= 0xc4ceb9fe1a85ec53L;
    hash ^= hash >> 33;

    hash &= (1ul << 32) - 1;
    return 1.0 * hash / ((1ul << 32) - 1) < probability;
  };

  auto sample_edge = [&](const NodeID u, EdgeWeight /* w */, const NodeID v) {
    return throw_dice(u, v);
  };

  return contract_and_sparsify_clustering(
      current().concretize<CSRGraph>(),
      std::move(mapping),
      c_n,
      sample_edge,
      _c_ctx.contraction,
      _contraction_m_ctx
  );
}

} // namespace kaminpar::shm
