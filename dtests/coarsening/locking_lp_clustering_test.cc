/*******************************************************************************
 * @file:   locking_lp_clustering_test.cc
 *
 * @author: Daniel Seemaier
 * @date:   04.10.21
 * @brief:  Unit tests for distributed locking label propagation clustering.
 ******************************************************************************/
#include "dkaminpar/coarsening/locking_lp_clustering.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dtests/mpi_test.h"

using ::testing::AnyOf;
using ::testing::Each;

namespace dkaminpar::test {
//  0---1-#-3---4
//  |\ /  #  \ /|
//  | 2---#---5 |
//  |  \  #  /  |
// ###############
//  |    \ /    |
//  |     8     |
//  |    / \    |
//  +---7---6---+
class DistributedTrianglesFixture : public DistributedGraphFixture {
protected:
  void SetUp() override {
    DistributedGraphFixture::SetUp();

    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
    ALWAYS_ASSERT(size == 3) << "must be tested on three PEs";

    scalable_vector<GlobalNodeID> node_distribution{0, 3, 6, 9};
    const GlobalNodeID global_n = 9;
    const GlobalEdgeID global_m = 30;

    n0 = 3 * rank;
    graph = dkaminpar::graph::Builder{}
                .initialize(global_n, global_m, rank, std::move(node_distribution))
                .create_node(1)
                .create_edge(1, n0 + 1)
                .create_edge(1, n0 + 2)
                .create_edge(1, prev(n0, 2, 9))
                .create_node(1)
                .create_edge(1, n0)
                .create_edge(1, n0 + 2)
                .create_edge(1, next(n0 + 1, 2, 9))
                .create_node(1)
                .create_edge(1, n0)
                .create_edge(1, n0 + 1)
                .create_edge(1, next(n0 + 2, 3, 9))
                .create_edge(1, prev(n0 + 2, 3, 9))
                .finalize();
  }

  DistributedGraph graph;
  GlobalNodeID n0;
};

auto compute_clustering(const DistributedGraph &graph, NodeWeight max_cluster_weight = 0) {
  // 0 --> no weight constraint
  if (max_cluster_weight == 0) { max_cluster_weight = graph.total_node_weight(); }

  Context ctx = create_default_context();
  LockingLpClustering algorithm(graph.n(), graph.total_n(), ctx.coarsening);
  return algorithm.compute_clustering(graph, max_cluster_weight);
}

TEST_F(DistributedTrianglesFixture, TestLocalClustering) {
  static constexpr EdgeWeight kInfinity = 100;
  // make internal edge much more attractive for contraction
  graph = graph::change_edge_weights_by_global_endpoints(std::move(graph), {{n0, n0 + 1, kInfinity},
                                                                            {n0 + 1, n0 + 2, kInfinity},
                                                                            {n0, n0 + 2, kInfinity}});

  // clustering should place all owned nodes into the same cluster, with a local node ID
  const auto &clustering = compute_clustering(graph);
  LOG << clustering;
  EXPECT_THAT(clustering, AnyOf(Each(0), Each(1), Each(2)));
}
} // namespace dkaminpar::test
