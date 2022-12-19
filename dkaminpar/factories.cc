/*******************************************************************************
 * @file:   factories.cc
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Instantiates the configured partitioning components.
 ******************************************************************************/
#include "dkaminpar/factories.h"

#include <memory>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/definitions.h"

// Partitioning schemes
#include "dkaminpar/partitioning/deep_multilevel_partitioner.h"
#include "dkaminpar/partitioning/kway_partitioner.h"

// Initial Partitioning
#include "dkaminpar/initial_partitioning/kaminpar_initial_partitioner.h"
#include "dkaminpar/initial_partitioning/mtkahypar_initial_partitioner.h"
#include "dkaminpar/initial_partitioning/random_initial_partitioner.h"

// Refinement
#include "dkaminpar/refinement/colored_lp_refiner.h"
#include "dkaminpar/refinement/fm_refiner.h"
#include "dkaminpar/refinement/greedy_balancer.h"
#include "dkaminpar/refinement/local_fm_refiner.h"
#include "dkaminpar/refinement/lp_refiner.h"
#include "dkaminpar/refinement/multi_refiner.h"
#include "dkaminpar/refinement/noop_refiner.h"

// Clustering
#include "dkaminpar/coarsening/global_active_set_label_propagation_clustering.h"
#include "dkaminpar/coarsening/global_label_propagation_clustering.h"
#include "dkaminpar/coarsening/hem_clustering.h"
#include "dkaminpar/coarsening/local_label_propagation_clustering.h"
#include "dkaminpar/coarsening/locking_label_propagation_clustering.h"
#include "dkaminpar/coarsening/noop_clustering.h"

namespace kaminpar::dist::factory {
std::unique_ptr<Partitioner> create_partitioner(const Context& ctx, const DistributedGraph& graph) {
    switch (ctx.partition.mode) {
        case PartitioningMode::DEEP:
            return std::make_unique<DeepMultilevelPartitioner>(graph, ctx);

        case PartitioningMode::KWAY:
            return std::make_unique<KWayPartitioner>(graph, ctx);
    }

    __builtin_unreachable();
}

std::unique_ptr<InitialPartitioner> create_initial_partitioning_algorithm(const Context& ctx) {
    switch (ctx.initial_partitioning.algorithm) {
        case InitialPartitioningAlgorithm::KAMINPAR:
            return std::make_unique<KaMinParInitialPartitioner>(ctx);

        case InitialPartitioningAlgorithm::MTKAHYPAR:
            return std::make_unique<MtKaHyParInitialPartitioner>(ctx);

        case InitialPartitioningAlgorithm::RANDOM:
            return std::make_unique<RandomInitialPartitioner>();
    }

    __builtin_unreachable();
}

namespace {
std::unique_ptr<Refiner> create_refinement_algorithm(const Context& ctx, const KWayRefinementAlgorithm algorithm) {
    switch (algorithm) {
        case KWayRefinementAlgorithm::NOOP:
            return std::make_unique<NoopRefiner>();

        case KWayRefinementAlgorithm::LP:
            return std::make_unique<LPRefiner>(ctx);

        case KWayRefinementAlgorithm::LOCAL_FM:
            return std::make_unique<LocalFMRefiner>(ctx);

        case KWayRefinementAlgorithm::FM:
            return std::make_unique<FMRefiner>(ctx);

        case KWayRefinementAlgorithm::COLORED_LP:
            return std::make_unique<ColoredLPRefiner>(ctx);

        case KWayRefinementAlgorithm::GREEDY_BALANCER:
            return std::make_unique<GreedyBalancer>(ctx);
    }

    __builtin_unreachable();
}
} // namespace

std::unique_ptr<Refiner> create_refinement_algorithm(const Context& ctx) {
    if (ctx.refinement.algorithms.size() == 1) {
        return create_refinement_algorithm(ctx, ctx.refinement.algorithms.front());
    }

    std::vector<std::unique_ptr<Refiner>> refiners;
    for (const KWayRefinementAlgorithm algorithm: ctx.refinement.algorithms) {
        refiners.push_back(create_refinement_algorithm(ctx, algorithm));
    }
    return std::make_unique<MultiRefiner>(std::move(refiners));
}

std::unique_ptr<ClusteringAlgorithm<GlobalNodeID>> create_global_clustering_algorithm(const Context& ctx) {
    switch (ctx.coarsening.global_clustering_algorithm) {
        case GlobalClusteringAlgorithm::NOOP:
            return std::make_unique<GlobalNoopClustering>(ctx);

        case GlobalClusteringAlgorithm::LP:
            return std::make_unique<DistributedGlobalLabelPropagationClustering>(ctx);

        case GlobalClusteringAlgorithm::ACTIVE_SET_LP:
            return std::make_unique<DistributedActiveSetGlobalLabelPropagationClustering>(ctx);

        case GlobalClusteringAlgorithm::LOCKING_LP:
            return std::make_unique<LockingLabelPropagationClustering>(ctx);

        case GlobalClusteringAlgorithm::HEM:
            return std::make_unique<HEMClustering>(ctx);
    }

    __builtin_unreachable();
}

std::unique_ptr<ClusteringAlgorithm<NodeID>> create_local_clustering_algorithm(const Context& ctx) {
    switch (ctx.coarsening.local_clustering_algorithm) {
        case LocalClusteringAlgorithm::NOOP:
            return std::make_unique<LocalNoopClustering>(ctx);

        case LocalClusteringAlgorithm::LP:
            return std::make_unique<DistributedLocalLabelPropagationClustering>(ctx);
    }

    __builtin_unreachable();
}
} // namespace kaminpar::dist::factory
