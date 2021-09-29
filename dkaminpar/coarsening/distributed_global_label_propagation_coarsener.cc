/*******************************************************************************
 * @file:   distributed_global_label_propagation_coarsener.cc
 *
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Label propagation across PEs, synchronizes cluster weights after
 * every weight, otherwise moves nodes without communication causing violations
 * of the balance constraint.
 ******************************************************************************/
#include "dkaminpar/coarsening/distributed_global_label_propagation_coarsener.h"

namespace dkaminpar {
const clustering::AtomicClusterArray<GlobalNodeID> &
DistributedGlobalLabelPropagationClustering::cluster(const DistributedGraph &graph,
                                                     const NodeWeight max_cluster_weight) {
  initialize(&graph, graph.total_n());
  _max_cluster_weight = max_cluster_weight;

  for (std::size_t iteration = 0; iteration < _max_num_iterations; ++iteration) {
    if (perform_iteration() == 0) { break; }
  }

  return clusters();
}
} // namespace dkaminpar