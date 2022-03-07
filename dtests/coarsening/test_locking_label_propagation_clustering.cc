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
using ::testing::Eq;

namespace dkaminpar::test {
using namespace fixtures3PE;

using Clustering = LockingLpClustering::AtomicClusterArray;

Clustering compute_clustering(const DistributedGraph &graph, NodeWeight max_cluster_weight = 0,
                              const std::size_t num_iterations = 1, const std::size_t num_chunks = 0) {
  // 0 --> no weight constraint
  if (max_cluster_weight == 0) {
    max_cluster_weight = std::numeric_limits<NodeWeight>::max();
  }

  Context ctx = create_default_context();
  ctx.setup(graph);
  ctx.coarsening.global_lp.num_iterations = num_iterations;
  if (num_chunks != 0) {
    ctx.coarsening.global_lp.num_chunks = num_chunks;
  }

  DLOG << V(graph.n()) << V(graph.total_n());

  LockingLpClustering algorithm(ctx);
  return algorithm.compute_clustering(graph, max_cluster_weight); // create copy
}

Clustering get_local_clustering(const DistributedGraph &graph, const Clustering &clustering) {
  Clustering local_clustering(graph.n());
  for (const NodeID u : graph.nodes()) {
    local_clustering[u] = clustering[u];
  }
  return local_clustering;
}

auto get_ghost_clustering(const DistributedGraph &graph, const Clustering &clustering) {
  std::unordered_map<PEID, std::vector<GlobalNodeID>> labels_on_pe;
  for (NodeID u : graph.ghost_nodes()) {
    labels_on_pe[graph.ghost_owner(u)].push_back(clustering[u]);
  }
  return labels_on_pe;
}

TEST_F(DistributedTriangles, TestLocalClustering) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+

  SINGLE_THREADED_TEST;

  static constexpr EdgeWeight kInfinity = 100;
  // make internal edge much more attractive for contraction
  graph = graph::change_edge_weights_by_global_endpoints(std::move(graph),
                                                         {{n0, n0 + 2, kInfinity}, {n0 + 1, n0 + 2, kInfinity}});

  // clustering should place all owned nodes into the same cluster, with a local node ID
  const auto clustering = compute_clustering(graph);
  std::vector<GlobalNodeID> local_clustering{clustering[0], clustering[1], clustering[2]};
  EXPECT_THAT(local_clustering, AnyOf(Each(0), Each(1), Each(2), Each(3), Each(4), Each(5), Each(6), Each(7), Each(8)));

  SLOG;
}

TEST_F(DistributedTriangles, TestGhostNodeLabelsAfterLocalClustering) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+

  SINGLE_THREADED_TEST;

  static constexpr EdgeWeight kInfinity = 100;
  // make internal edge much more attractive for contraction
  graph = graph::change_edge_weights_by_global_endpoints(std::move(graph),
                                                         {{n0, n0 + 2, kInfinity}, {n0 + 1, n0 + 2, kInfinity}});

  // clustering should place all owned nodes into the same cluster, with a local node ID
  const auto clustering = compute_clustering(graph);

  std::unordered_map<PEID, std::vector<GlobalNodeID>> labels_on_pe;
  for (NodeID u : graph.ghost_nodes()) {
    labels_on_pe[graph.ghost_owner(u)].push_back(clustering[u]);
  }

  for (const auto &[pe, labels] : labels_on_pe) {
    EXPECT_THAT(labels, AnyOf(Each(0), Each(1), Each(2), Each(3), Each(4), Each(5), Each(6), Each(7), Each(8)));
  }
}

TEST_F(DistributedTriangles, TestLocalClusteringWithTwoIterations) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+

  SINGLE_THREADED_TEST;

  static constexpr EdgeWeight kInfinity = 100;
  // add path of increasing weights
  graph = graph::change_edge_weights_by_global_endpoints(std::move(graph),
                                                         {{n0, n0 + 1, kInfinity}, {n0 + 1, n0 + 2, 2 * kInfinity}});

  { // make one iteration
    const auto clustering = compute_clustering(graph);
    EXPECT_THAT(clustering[0], Eq(n0 + 1));
    EXPECT_THAT(clustering[1], Eq(n0 + 2));
    EXPECT_THAT(clustering[2], Eq(n0 + 2));
  }
  { // make two iterations
    const auto clustering = compute_clustering(graph, 0, 2);
    EXPECT_THAT(clustering[0], Eq(n0 + 2));
    EXPECT_THAT(clustering[1], Eq(n0 + 2));
    EXPECT_THAT(clustering[2], Eq(n0 + 2));
  }
}

TEST_F(DistributedTriangles, TestLocalClusteringWithWeightConstraintAndTwoIterations) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+

  SINGLE_THREADED_TEST;

  static constexpr EdgeWeight kInfinity = 100;
  graph = graph::change_edge_weights_by_global_endpoints(std::move(graph),
                                                         {{n0, n0 + 1, kInfinity}, {n0 + 1, n0 + 2, 2 * kInfinity}});

  static constexpr NodeWeight kClusterWeightLimit = 2; // 1 node cannot join
  {                                                    // make one iteration
    const auto clustering = compute_clustering(graph, kClusterWeightLimit, 1);
    EXPECT_THAT(clustering[0], Eq(n0 + 1));
    EXPECT_THAT(clustering[1], Eq(n0 + 2));
    EXPECT_THAT(clustering[2], Eq(n0 + 2));
  }
  { // make two iterations
    const auto clustering = compute_clustering(graph, kClusterWeightLimit, 2);
    EXPECT_THAT(clustering[0], Eq(n0 + 1));
    EXPECT_THAT(clustering[1], Eq(n0 + 2));
    EXPECT_THAT(clustering[2], Eq(n0 + 2));
  }
}

// TEST_F(DistributedTriangles, TestConvergenceWithMultipleChunks) {
//   SINGLE_THREADED_TEST;
//
//   // no cluster weight limit, 10 iterations -> everything should be the same cluster
//   const auto clustering = compute_clustering(graph, 0, 10);
//   EXPECT_THAT(clustering, AnyOf(Each(0), Each(1), Each(2), Each(3), Each(4), Each(5), Each(6), Each(7), Each(8)));
// }

TEST_F(DistributedTriangles, TestGhostClusteringOneRequestPerChunkAndPE) {
  //   0---1=#=3---4
  //  ||\ /  #  \ /||
  //  || 2---#---5 ||
  //  ||  \  #  /  ||
  // #################
  //  ||    \ /    ||
  //  ||     8     ||
  //  ||    / \    ||
  //  ++===7---6===++

  SINGLE_THREADED_TEST;

  static constexpr EdgeWeight kInfinity = 100;

  int rank = mpi::get_comm_rank();
  std::vector<std::tuple<GlobalNodeID, GlobalNodeID, EdgeWeight>> changes;
  if (rank == 0) {
    changes.emplace_back(0, 7, kInfinity);
    changes.emplace_back(1, 3, kInfinity);
    changes.emplace_back(0, 2, kInfinity / 2);
  } else if (rank == 1) {
    changes.emplace_back(1, 3, kInfinity);
    changes.emplace_back(4, 6, kInfinity);
    changes.emplace_back(3, 5, kInfinity / 2);
  } else if (rank == 2) {
    changes.emplace_back(6, 4, kInfinity);
    changes.emplace_back(7, 0, kInfinity);
    changes.emplace_back(6, 8, kInfinity / 2);
  }
  graph = graph::change_edge_weights_by_global_endpoints(std::move(graph), changes);
  const auto clustering = compute_clustering(graph);

  // since all PEs have the same number of edges, smaller PEs should win the tie-breaking
  if (rank == 0) {
    EXPECT_THAT(clustering[0], Eq(7));
    EXPECT_THAT(clustering[1], Eq(1));
    EXPECT_THAT(clustering[2], Eq(7));
  } else if (rank == 1) {
    EXPECT_THAT(clustering[0], Eq(1));
    EXPECT_THAT(clustering[1], Eq(4));
    EXPECT_THAT(clustering[2], Eq(1)); // if 5 is not in the same chunk as 3
  } else if (rank == 2) {
    EXPECT_THAT(clustering[0], Eq(4));
    EXPECT_THAT(clustering[1], Eq(7));
    EXPECT_THAT(clustering[2], Eq(4)); // if 8 is not in the same chunk as 6
  }
}

TEST_F(DistributedPathOneNodePerPE, TestGhostClusteringAlongPathWithTwoIterations) {
  // 0-#-1=#=2
  SINGLE_THREADED_TEST;

  int rank = mpi::get_comm_rank();
  std::vector<std::tuple<GlobalNodeID, GlobalNodeID, EdgeWeight>> changes;
  if (rank == 0) { // edge weights increase along path
    changes.emplace_back(0, 1, 2);
  } else if (rank == 1) {
    changes.emplace_back(1, 0, 2);
    changes.emplace_back(1, 2, 4);
  } else if (rank == 2) {
    changes.emplace_back(2, 1, 4);
  }
  graph = graph::change_edge_weights_by_global_endpoints(std::move(graph), changes);
  graph.print();

  {
    const auto clustering = compute_clustering(graph);
    SLOG << "1 iteration: " << V(clustering);
    if (rank == 0) {
      EXPECT_THAT(clustering[0], Eq(0)); // 0 rejected because 1 tried to move to another PE
    } else {
      EXPECT_THAT(clustering[0], Eq(2));
    }
  }
  {
    const auto clustering = compute_clustering(graph, 0, 2);
    EXPECT_THAT(clustering[0], Eq(2));
  }

  mpi::barrier(MPI_COMM_WORLD);
}
} // namespace dkaminpar::test
