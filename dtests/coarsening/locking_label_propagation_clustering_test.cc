/*******************************************************************************
 * @file:   locking_lp_clustering_test.cc
 *
 * @author: Daniel Seemaier
 * @date:   04.10.21
 * @brief:  Unit tests for distributed locking label propagation clustering.
 ******************************************************************************/
#include "dkaminpar/coarsening/locking_label_propagation_clustering.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dtests/mpi_test.h"

#include <tbb/global_control.h>
#include <unordered_map>

using ::testing::AnyOf;
using ::testing::Each;

namespace dkaminpar::test {
using namespace fixtures3PE;

auto compute_clustering(const DistributedGraph &graph, NodeWeight max_cluster_weight = 0,
                        const std::size_t num_iterations = 1) {
  // 0 --> no weight constraint
  if (max_cluster_weight == 0) { max_cluster_weight = graph.total_node_weight(); }

  Context ctx = create_default_context();
  ctx.coarsening.lp.num_iterations = num_iterations;

  DLOG << V(graph.n()) << V(graph.total_n());

  LockingLpClustering algorithm(graph.n(), graph.total_n(), ctx.coarsening);
  return algorithm.compute_clustering(graph, max_cluster_weight);
}

TEST_F(DistributedTriangles, TestLocalClustering) {
  auto gc = tbb::global_control{tbb::global_control::max_allowed_parallelism, 1};

  static constexpr EdgeWeight kInfinity = 100;
  // make internal edge much more attractive for contraction
  graph = graph::change_edge_weights_by_global_endpoints(std::move(graph), {{n0, n0 + 1, kInfinity},
                                                                            {n0 + 1, n0 + 2, kInfinity},
                                                                            {n0, n0 + 2, kInfinity}});

  // clustering should place all owned nodes into the same cluster, with a local node ID
  const auto &clustering = compute_clustering(graph);
  std::vector<GlobalNodeID> local_clustering{clustering[0], clustering[1], clustering[2]};
  EXPECT_THAT(local_clustering, AnyOf(Each(0), Each(1), Each(2), Each(3), Each(4), Each(5), Each(6), Each(7), Each(8)));
}

TEST_F(DistributedTriangles, TestGhostNodeLabelsAfterLocalClustering) {
  auto gc = tbb::global_control{tbb::global_control::max_allowed_parallelism, 1};

  static constexpr EdgeWeight kInfinity = 100;
  // make internal edge much more attractive for contraction
  graph = graph::change_edge_weights_by_global_endpoints(std::move(graph), {{n0, n0 + 1, kInfinity},
                                                                            {n0 + 1, n0 + 2, kInfinity},
                                                                            {n0, n0 + 2, kInfinity}});

  // clustering should place all owned nodes into the same cluster, with a local node ID
  const auto &clustering = compute_clustering(graph);

  std::unordered_map<PEID, std::vector<GlobalNodeID>> labels_on_pe;
  for (NodeID u : graph.ghost_nodes()) { labels_on_pe[graph.ghost_owner(u)].push_back(clustering[u]); }

  for (const auto &[pe, labels] : labels_on_pe) {
    EXPECT_THAT(labels, AnyOf(Each(0), Each(1), Each(2), Each(3), Each(4), Each(5), Each(6), Each(7), Each(8)));
  }
}
} // namespace dkaminpar::test
