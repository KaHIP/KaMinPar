/*******************************************************************************
 * @file:   locking_clustering_contraction.cc
 *
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 * @brief:  Contracts a clustering computed by \c LockingLabelPropagation.
 ******************************************************************************/
#include "dkaminpar/algorithm/locking_clustering_contraction.h"

#include "dkaminpar/mpi_graph_utils.h"

#include <unordered_set>

namespace dkaminpar::graph {
namespace {
#ifdef KAMINPAR_ENABLE_ASSERTIONS
bool CHECK_CLUSTERING_INVARIANT(const DistributedGraph &graph,
                                const LockingLpClustering::AtomicClusterArray &clustering) {
  ASSERT(graph.n() <= clustering.size());

  // set of nonempty labels on this PE
  std::unordered_set<GlobalNodeID> nonempty_labels;
  for (const NodeID u : graph.nodes()) { nonempty_labels.insert(clustering[u]); }

  mpi::graph::sparse_alltoall_custom<GlobalNodeID>(
      graph, 0, graph.n(),
      [&](const NodeID u) {
        ASSERT(clustering[u] < graph.global_n());
        return !graph.is_owned_global_node(clustering[u]);
      },
      [&](const NodeID u) { return std::make_pair(clustering[u], graph.find_owner_of_global_node(clustering[u])); },
      [&](const auto &buffer, const PEID pe) {
        for (const GlobalNodeID label : buffer) {
          ASSERT(nonempty_labels.contains(label))
              << label << " from PE " << pe << " does not exist on PE " << mpi::get_comm_rank(MPI_COMM_WORLD);
        }
      });

  return true;
}
#endif // KAMINPAR_ENABLE_ASSERTIONS
} // namespace

contraction::LockingClusteringContractionResult
contract_locking_clustering(const DistributedGraph &graph,
                                        const LockingLpClustering::AtomicClusterArray &clustering,
                                        contraction::MemoryContext m_ctx) {
  ASSERT(CHECK_CLUSTERING_INVARIANT(graph, clustering));
  return {{}, {}, std::move(m_ctx)};
}
} // namespace dkaminpar::graph
