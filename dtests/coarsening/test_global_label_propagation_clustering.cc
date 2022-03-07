/*******************************************************************************
 * @file:   global_label_propagation_clustering_test.cc
 *
 * @author: Daniel Seemaier
 * @date:   22.11.2021
 * @brief:  Unit tests for global LP clustering.
 ******************************************************************************/
#include "dtests/mpi_test.h"

#include "dkaminpar/coarsening/global_label_propagation_clustering.h"

using ::testing::AnyOf;
using ::testing::Each;
using ::testing::Eq;

namespace dkaminpar::test {
using namespace fixtures3PE;

auto compute_clustering(const DistributedGraph &graph, const NodeWeight max_cluster_weight, bool merge_singleton_nodes,
                        bool single_round = false) {
  Context ctx = ::dkaminpar::create_default_context();
  ctx.coarsening.global_lp.merge_singleton_clusters = merge_singleton_nodes;
  ctx.coarsening.global_lp.num_iterations = 5;
  if (single_round) {
    ctx.coarsening.global_lp.num_chunks = 1;
  }
  ctx.setup(graph);

  ::dkaminpar::DistributedGlobalLabelPropagationClustering clustering(ctx);
  return clustering.compute_clustering(graph, max_cluster_weight);
}

TEST_F(DistributedTriangles, HeavyEdgesOnPEs) {
  //  0====1-#-3====4
  //  |\\//  #  \\//|
  //  | 2----#----5 |
  //  |  \   #   / |
  // ###############
  //  |     \ /    |
  //  |      8     |
  //  |    //\\    |
  //  +---7====6---+
  graph = graph::change_edge_weights_by_endpoints(std::move(graph), {
                                                                        {0, 1, 10},
                                                                        {0, 2, 5},
                                                                        {1, 2, 5},
                                                                    });

  auto clustering = compute_clustering(graph, 3, true);

  // owned nodes should be in a single cluster
  std::vector<Atomic<GlobalNodeID>> owned_clustering{clustering[0], clustering[1], clustering[2]};
  EXPECT_THAT(owned_clustering, AnyOf(Each(rank * 3), Each(rank * 3 + 1), Each(rank * 3 + 2)));

  // gather ghost node labels on the same PE
  std::vector<std::vector<GlobalNodeID>> ghost_node_labels(3);
  for (NodeID u = graph.n(); u < graph.total_n(); ++u) {
    ghost_node_labels[graph.ghost_owner(u)].push_back(clustering[u]);
  }

  // ghost nodes on the same PE should be in a single cluster
  for (PEID pe = 0; pe < 3; ++pe) {
    EXPECT_THAT(ghost_node_labels[pe], AnyOf(Each(pe * 3), Each(pe * 3 + 1), Each(pe * 3 + 2)));
  }
}

// If isolated nodes should not be matched, they should still reside in their initial singleton cluster
TEST_F(DistributedGraphWith900NodesAnd0Edges, KeepsIsolatedNodesSeparated) {
  auto clustering = compute_clustering(graph, 300, false);
  for (const NodeID u : graph.nodes()) {
    EXPECT_EQ(clustering[u], graph.local_to_global_node(u));
  }
}

// If isolated nodes are matched, they should build a single cluster on each PE
TEST_F(DistributedGraphWith900NodesAnd0Edges, MergesIsolatedNodesSingleThreaded) {
  // on a single thread and round, all nodes should be in the same cluster
  SINGLE_THREADED_TEST;
  auto clustering = compute_clustering(graph, 300, true, true);
  EXPECT_THAT(clustering, Each(clustering.front()));
}
} // namespace dkaminpar::test
