/*******************************************************************************
 * @file:   contraction_test.cc
 * @author: Daniel Seemaier
 * @date:   28.01.2023
 * @brief:  Unit tests for the graph contraction function.
 ******************************************************************************/
#include <gmock/gmock.h>

#include "tests/dkaminpar/distributed_graph_factories.h"
#include "tests/dkaminpar/distributed_graph_helpers.h"

#include "dkaminpar/coarsening/contraction/cluster_contraction.h"
#include "dkaminpar/mpi/utils.h"

namespace kaminpar::dist {
using namespace kaminpar::dist::testing;
using namespace ::testing;

namespace {
StaticArray<GlobalNodeID> build_cnode_distribution(const GlobalNodeID n) {
  return mpi::build_distribution_from_local_count<GlobalNodeID, StaticArray>(n, MPI_COMM_WORLD);
}
} // namespace

TEST(ClusterReassignmentTest, perfectly_balanced_case) {
  const auto graph = make_isolated_nodes_graph(4);
  const auto cnode_distribution = build_cnode_distribution(2);
  const auto result = compute_assignment_shifts(graph, cnode_distribution, 1.0);
  EXPECT_THAT(result.overload, Each(Eq(0)));
  EXPECT_THAT(result.underload, Each(Eq(0)));
}

TEST(ClusterReassignmentTest, stair_no_limit) {
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  const auto graph = make_isolated_nodes_graph(size * size);
  const auto cnode_distribution = build_cnode_distribution(2 * (rank + 1));
  const auto result = compute_assignment_shifts(graph, cnode_distribution, 1.0);

  const GlobalNodeID expected = size + 1;

  for (PEID pe = 0; pe < size / 2; ++pe) {
    EXPECT_EQ(result.overload[pe + 1] - result.overload[pe], 0);
    EXPECT_EQ(result.underload[pe + 1] - result.underload[pe], expected - 2 * (pe + 1));
  }
  for (PEID pe = size / 2; pe < size; ++pe) {
    EXPECT_EQ(result.overload[pe + 1] - result.overload[pe], 2 * (pe + 1) - expected);
    EXPECT_EQ(result.underload[pe + 1] - result.underload[pe], 0);
  }
}

TEST(ClusterContractionTest, contract_empty_graph) {
  auto graph = make_empty_graph();

  GlobalClustering clustering;
  auto [c_graph, c_mapping, migration] = contract_clustering(graph, clustering);

  EXPECT_EQ(c_graph.n(), 0);
  EXPECT_EQ(c_graph.global_n(), 0);
  EXPECT_EQ(c_graph.m(), 0);
  EXPECT_EQ(c_graph.global_m(), 0);

  EXPECT_TRUE(c_mapping.empty());
}

TEST(ClusterContractionTest, contract_local_edge) {
  const auto graph = make_isolated_edges_graph(1);
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  GlobalClustering clustering{graph.offset_n(), graph.offset_n()};
  auto [c_graph, c_mapping, migration] = contract_clustering(graph, clustering);

  EXPECT_EQ(c_graph.global_n(), size);
  EXPECT_EQ(c_graph.m(), 0);
  EXPECT_EQ(c_graph.global_m(), 0);
  ASSERT_EQ(c_graph.n(), 1);
  EXPECT_EQ(c_graph.node_weight(0), 2);

  EXPECT_THAT(c_mapping, Each(Eq(c_graph.offset_n())));
}

TEST(ClusterContractionTest, contract_local_complete_graph) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  for (const NodeID clique_size : {1, 5, 10}) {
    const auto graph = make_local_complete_graph(clique_size);

    GlobalClustering clustering(clique_size, graph.offset_n());
    auto [c_graph, c_mapping, migration] = contract_clustering(graph, clustering);

    EXPECT_EQ(c_graph.global_n(), size);
    EXPECT_EQ(c_graph.global_m(), 0);
    EXPECT_EQ(c_graph.m(), 0);
    ASSERT_EQ(c_graph.n(), 1);
    EXPECT_EQ(c_graph.node_weight(0), static_cast<NodeWeight>(clique_size));

    EXPECT_THAT(c_mapping, Each(Eq(c_graph.offset_n())));
  }
}

TEST(ClusterContractionTest, contract_local_complete_bipartite_graph_vertically) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  for (const NodeID set_size : {1, 5, 10}) {
    const auto graph = make_local_complete_bipartite_graph(set_size);

    GlobalClustering clustering(2 * set_size);
    std::fill(clustering.begin(), clustering.begin() + set_size, graph.offset_n());
    std::fill(clustering.begin() + set_size, clustering.end(), graph.offset_n() + set_size);
    auto [c_graph, c_mapping, migration] = contract_clustering(graph, clustering);

    EXPECT_EQ(c_graph.global_n(), 2 * size);
    EXPECT_EQ(c_graph.global_m(), 2 * size);

    ASSERT_EQ(c_graph.n(), 2);
    EXPECT_EQ(c_graph.node_weight(0), set_size);
    EXPECT_EQ(c_graph.node_weight(1), set_size);

    ASSERT_EQ(c_graph.m(), 2);
    EXPECT_EQ(c_graph.edge_weight(0), set_size * set_size);
    EXPECT_EQ(c_graph.edge_weight(1), set_size * set_size);

    ASSERT_EQ(c_mapping.size(), graph.n());

    EXPECT_THAT(c_mapping[0], AnyOf(Eq(c_graph.offset_n()), Eq(c_graph.offset_n() + 1)));
    for (NodeID u = 0; u < set_size; ++u) {
      EXPECT_EQ(c_mapping[0], c_mapping[u]);
    }

    EXPECT_NE(c_mapping[0], c_mapping[set_size]);

    EXPECT_THAT(c_mapping[set_size], AnyOf(Eq(c_graph.offset_n()), Eq(c_graph.offset_n() + 1)));
    for (NodeID u = set_size; u < 2 * set_size; ++u) {
      EXPECT_EQ(c_mapping[set_size], c_mapping[u]);
    }
  }
}

TEST(ClusterContractionTest, contract_local_complete_bipartite_graph_horizontally) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);

  for (const NodeID set_size : {1, 5, 10}) {
    const auto graph = make_local_complete_bipartite_graph(set_size);

    GlobalClustering clustering(2 * set_size);
    std::iota(clustering.begin(), clustering.end(), 0u);
    std::transform(
        clustering.begin(),
        clustering.end(),
        clustering.begin(),
        [&](const GlobalNodeID value) { return graph.offset_n() + value % set_size; }
    );
    auto [c_graph, c_mapping, migration] = contract_clustering(graph, clustering);

    EXPECT_EQ(c_graph.global_n(), set_size * size) << "Set size: " << set_size;
    EXPECT_EQ(c_graph.global_m(), set_size * (set_size - 1) * size) << "Set size: " << set_size;

    ASSERT_EQ(c_graph.n(), set_size);
    EXPECT_THAT(c_graph.node_weights(), Each(Eq(2)));

    ASSERT_EQ(c_graph.m(), set_size * (set_size - 1));
    EXPECT_THAT(c_graph.edge_weights(), Each(Eq(2)));

    ASSERT_EQ(c_mapping.size(), graph.n());
    EXPECT_THAT(
        c_mapping, Each(AllOf(Lt(c_graph.offset_n() + c_graph.n()), Ge(c_graph.offset_n())))
    );

    for (NodeID u = 0; u < set_size; ++u) {
      EXPECT_EQ(c_mapping[u], c_mapping[u + set_size]);
      EXPECT_THAT(c_mapping, Contains(c_mapping[u]).Times(2));
    }
  }
}

TEST(ClusterContractionTest, contract_global_complete_graph_to_single_node) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  for (const NodeID nodes_per_pe : {5}) {
    const auto graph = make_global_complete_graph(nodes_per_pe);
    GlobalClustering clustering(graph.total_n(), 0);
    const auto [c_graph, c_mapping, migration] = contract_clustering(graph, clustering);

    EXPECT_EQ(c_graph.global_n(), 1);
    EXPECT_EQ(c_graph.global_m(), 0);
    EXPECT_EQ(c_graph.m(), 0);

    if (rank == 0) {
      ASSERT_EQ(c_graph.n(), 1);
      EXPECT_EQ(c_graph.node_weight(0), nodes_per_pe * size);
    } else {
      EXPECT_EQ(c_graph.n(), 0);
    }

    EXPECT_EQ(c_mapping.size(), graph.n());
    EXPECT_THAT(c_mapping, Each(Eq(0)));
  }
}

TEST(ClusterContractionTest, contract_global_complete_graph_to_one_node_per_pe) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  for (const NodeID nodes_per_pe : {1, 5, 10}) {
    const auto graph = make_global_complete_graph(nodes_per_pe);
    GlobalClustering clustering(graph.total_n());
    graph.pfor_all_nodes([&](const NodeID u) {
      const PEID pe = graph.is_owned_node(u) ? rank : graph.ghost_owner(u);
      clustering[u] = graph.offset_n(pe);
    });
    const auto [c_graph, c_mapping, migration] = contract_clustering(graph, clustering);

    EXPECT_EQ(c_graph.global_n(), size);
    EXPECT_EQ(c_graph.global_m(), size * (size - 1));

    ASSERT_EQ(c_graph.n(), 1);
    ASSERT_EQ(c_graph.m(), size - 1);

    EXPECT_EQ(c_graph.node_weight(0), nodes_per_pe);
    EXPECT_THAT(c_graph.edge_weights(), Each(Eq(nodes_per_pe * nodes_per_pe)));

    ASSERT_EQ(c_mapping.size(), graph.n());
    EXPECT_THAT(c_mapping, Each(Eq(c_graph.offset_n())));
  }
}

TEST(ClusterContractionTest, keep_global_complete_graph) {
  for (const NodeID nodes_per_pe : {1, 5, 10}) {
    const auto graph = make_global_complete_graph(nodes_per_pe);
    GlobalClustering clustering(graph.total_n());
    graph.pfor_all_nodes([&](const NodeID u) { clustering[u] = graph.local_to_global_node(u); });
    const auto [c_graph, c_mapping, migration] = contract_clustering(graph, clustering);

    EXPECT_EQ(c_graph.global_n(), graph.global_n());
    EXPECT_EQ(c_graph.global_m(), graph.global_m());
    EXPECT_EQ(c_graph.n(), graph.n());
    EXPECT_EQ(c_graph.m(), graph.m());
    EXPECT_EQ(c_graph.node_weights(), graph.node_weights());
    EXPECT_EQ(c_graph.edge_weights(), graph.edge_weights());

    ASSERT_EQ(c_mapping.size(), graph.n());
    for (NodeID u = 0; u < graph.n(); ++u) {
      EXPECT_THAT(
          c_mapping[u], AllOf(Ge(c_graph.offset_n()), Lt(c_graph.offset_n() + c_graph.n()))
      );
      EXPECT_THAT(c_mapping, Contains(c_mapping[u]).Times(1));
    }
  }
}

TEST(ClusterContractionTest, rotate_global_complete_graph) {
  for (const NodeID nodes_per_pe : {1, 5, 10}) {
    const auto graph = make_global_complete_graph(nodes_per_pe);
    GlobalClustering clustering(graph.total_n());
    graph.pfor_all_nodes([&](const NodeID u) {
      clustering[u] = (graph.local_to_global_node(u) + 1) % graph.global_n();
    });
    const auto [c_graph, c_mapping, migration] = contract_clustering(graph, clustering);

    EXPECT_EQ(c_graph.global_n(), graph.global_n());
    EXPECT_EQ(c_graph.global_m(), graph.global_m());
    EXPECT_EQ(c_graph.n(), graph.n());
    EXPECT_EQ(c_graph.m(), graph.m());
    EXPECT_EQ(c_graph.node_weights(), graph.node_weights());
    EXPECT_EQ(c_graph.edge_weights(), graph.edge_weights());
  }
}
} // namespace kaminpar::dist
