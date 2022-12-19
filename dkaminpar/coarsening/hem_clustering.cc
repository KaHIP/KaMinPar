#include "dkaminpar/coarsening/hem_clustering.h"

namespace kaminpar::dist {
HEMClustering::HEMClustering(const Context& ctx) {
    ((void)ctx);
}

const HEMClustering::AtomicClusterArray&
HEMClustering::compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) {
    ((void)graph);
    ((void)max_cluster_weight);
    return _matching;
}
} // namespace kaminpar::dist
