/*******************************************************************************
 * @file:   factories.cc
 *
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Instantiates the configured partitioning components.
 ******************************************************************************/
#include "dkaminpar/factories.h"

// Initial Partitioning
#include "coarsening/local_label_propagation_clustering.h"
#include "definitions.h"
#include "dkaminpar/initial_partitioning/kaminpar_initial_partitioner.h"
#include "dkaminpar/initial_partitioning/random_initial_partitioner.h"

// Refinement
#include "dkaminpar/refinement/distributed_probabilistic_label_propagation_refiner.h"
#include "dkaminpar/refinement/noop_refiner.h"

// Clustering
#include "dkaminpar/coarsening/global_label_propagation_clustering.h"
#include "dkaminpar/coarsening/local_label_propagation_clustering.h"
#include "dkaminpar/coarsening/locking_label_propagation_clustering.h"
#include "dkaminpar/coarsening/noop_clustering.h"

namespace dkaminpar::factory {
std::unique_ptr<IInitialPartitioner> create_initial_partitioning_algorithm(const Context& ctx) {
    switch (ctx.initial_partitioning.algorithm) {
        case InitialPartitioningAlgorithm::KAMINPAR:
            return std::make_unique<KaMinParInitialPartitioner>(ctx);

        case InitialPartitioningAlgorithm::RANDOM:
            return std::make_unique<RandomInitialPartitioner>(ctx);
    }
    __builtin_unreachable();
}

std::unique_ptr<IDistributedRefiner> create_refinement_algorithm(const Context& ctx) {
    switch (ctx.refinement.algorithm) {
        case KWayRefinementAlgorithm::NOOP:
            return std::make_unique<NoopRefiner>();

        case KWayRefinementAlgorithm::PROB_LP:
            return std::make_unique<DistributedProbabilisticLabelPropagationRefiner>(ctx);
    }
    __builtin_unreachable();
}

std::unique_ptr<ClusteringAlgorithm<GlobalNodeID>> create_global_clustering_algorithm(const Context& ctx) {
    switch (ctx.coarsening.global_clustering_algorithm) {
        case GlobalClusteringAlgorithm::NOOP:
            return std::make_unique<GlobalNoopClustering>(ctx);

        case GlobalClusteringAlgorithm::LP:
            return std::make_unique<DistributedGlobalLabelPropagationClustering>(ctx);

        case GlobalClusteringAlgorithm::LOCKING_LP:
            return std::make_unique<LockingLabelPropagationClustering>(ctx);
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
} // namespace dkaminpar::factory
