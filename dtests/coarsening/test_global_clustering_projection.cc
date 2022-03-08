/*******************************************************************************
 * @file:   global_contraction_redistribution_test.cc
 *
 * @author: Daniel Seemaier
 * @date:   04.11.2021
 * @brief:  Unit tests for graph projects that do not make any assumptions on
 * how the contracted graph is distributed across PEs.
 ******************************************************************************/
#include "dkaminpar/coarsening/global_clustering_contraction.h"

#include "dtests/mpi_test.h"

using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::UnorderedElementsAre;

namespace dkaminpar::test {
using namespace fixtures3PE;

using Clustering = coarsening::GlobalClustering;

auto contract_clustering(const DistributedGraph &graph, const Clustering &clustering) {
  return coarsening::contract_global_clustering_full_migration(graph, clustering);
}

DistributedPartitionedGraph create_node_weight_partition(const DistributedGraph &graph) {
  scalable_vector<Atomic<BlockID>> partition(graph.total_n());
  scalable_vector<Atomic<BlockWeight>> block_weights(graph.global_n() + 1);
  for (const NodeID u : graph.all_nodes()) {
    partition[u] = graph.node_weight(u);
    block_weights[u] = graph.node_weight(u);
  }

  return {&graph, static_cast<BlockID>(graph.global_total_node_weight() + 1), std::move(partition),
          std::move(block_weights)};
}

void expect_partition(const DistributedPartitionedGraph &p_graph, const scalable_vector<BlockID> &expected_partition) {
  for (const NodeID u : p_graph.all_nodes()) {
    const BlockID expected_block = expected_partition[p_graph.local_to_global_node(u)];
    const BlockID actual_block = p_graph.block(u);
    EXPECT_EQ(actual_block, expected_block);
  }
}

TEST_F(DistributedTriangles, ProjectingFromSingletonClustersWorks) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |J
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+
  graph = graph::use_global_id_as_node_weight(std::move(graph));

  const auto clustering = graph::distribute_node_info<Clustering>(this->graph, {0, 1, 2, 3, 4, 5, 6, 7, 8});
  const auto [c_graph, c_mapping] = contract_clustering(this->graph, clustering);

  // place each node in a globally unique block
  auto pc_graph = create_node_weight_partition(c_graph);

  auto p_graph = coarsening::project_global_contracted_graph(graph, std::move(pc_graph), c_mapping);

  expect_partition(p_graph, {1, 2, 3, 4, 5, 6, 7, 8, 9});
}

TEST_F(DistributedTriangles, ProjectFromSingleNodeContractedOnPE0) {
  //  0---1-#-3---4
  //  |\ /  #  \ /|
  //  | 2---#---5 |
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |
  //  |    / \    |
  //  +---7---6---+
  graph = graph::use_global_id_as_node_weight(std::move(graph));

  auto [c_graph, c_mapping] = contract_clustering(graph, {0, 0, 0, 0, 0, 0, 0});
  auto pc_graph = create_node_weight_partition(c_graph);
  auto p_graph = coarsening::project_global_contracted_graph(graph, std::move(pc_graph), c_mapping);

  expect_partition(p_graph, {45, 45, 45, 45, 45, 45, 45, 45, 45});
}

TEST_F(DistributedTriangles, ProjectFromRowWiseContraction) {
  //  0---1-#-3---4  -- C0/W3 # C1/W9
  //  |\ /  #  \ /|
  //  | 2---#---5 |  -- C2/W9
  //  |  \  #  /  |
  // ###############
  //  |    \ /    |
  //  |     8     |  -- C3/W9
  //  |    / \    |
  //  +---7---6---+  -- C4/W15
  graph = graph::use_global_id_as_node_weight(std::move(graph));

  Clustering clustering = graph::distribute_node_info<Clustering>(graph, {0, 0, 2, 1, 1, 2, 4, 4, 3});
  const auto [c_graph, c_mapping] = contract_clustering(graph, clustering);
  auto pc_graph = create_node_weight_partition(c_graph);
  auto p_graph = coarsening::project_global_contracted_graph(graph, std::move(pc_graph), c_mapping);

  expect_partition(p_graph, {3, 3, 9, 9, 9, 9, 15, 15, 9});
}
} // namespace dkaminpar::test