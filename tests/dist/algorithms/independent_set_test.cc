/*******************************************************************************
 * @file:   independent_set_test.cc
 * @author: Daniel Seemaier
 * @date:   22.08.2022
 * @brief:  Tests for algorithms to find a independent set on distributed
 *graphs.
 ******************************************************************************/
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "tests/dist/distributed_graph_factories.h"
#include "tests/dist/distributed_graph_helpers.h"

#include "kaminpar-mpi/utils.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/algorithms/independent_set.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"

#include "kaminpar-common/asserting_cast.h"

namespace kaminpar::dist::graph {
using namespace kaminpar::dist::testing;

namespace {
template <typename Container>
std::vector<GlobalNodeID>
allgather_independent_set(const DistributedPartitionedGraph &p_graph, const Container &is) {
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);

  std::vector<int> recvcounts(size);
  recvcounts[rank] = asserting_cast<int>(is.size());
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> displs(size + 1);
  std::partial_sum(recvcounts.begin(), recvcounts.end(), displs.begin() + 1);

  std::vector<GlobalNodeID> global_is(displs.back());
  const int offset = displs[rank];
  for (std::size_t i = 0; i < is.size(); ++i) {
    global_is[offset + i] = p_graph.local_to_global_node(is[i]);
  }

  MPI_Allgatherv(
      MPI_IN_PLACE,
      0,
      MPI_DATATYPE_NULL,
      global_is.data(),
      recvcounts.data(),
      displs.data(),
      mpi::type::get<GlobalNodeID>(),
      MPI_COMM_WORLD
  );

  return global_is;
}

template <typename Container>
void expect_nonempty_independent_set(
    const DistributedPartitionedGraph &p_graph, const Container &is
) {
  const auto global_is = allgather_independent_set(p_graph, is);
  EXPECT_GE(global_is.size(), 0);

  std::vector<bool> is_in_independent_set(p_graph.total_n());
  for (const GlobalNodeID global_u : global_is) {
    if (p_graph.contains_global_node(global_u)) {
      is_in_independent_set[p_graph.global_to_local_node(global_u)] = true;
    }
  }

  for (const NodeID u : p_graph.nodes()) {
    if (!is_in_independent_set[u]) {
      continue;
    }

    for (const NodeID v : p_graph.adjacent_nodes(u)) {
      EXPECT_FALSE(is_in_independent_set[v]);
    }
  }
}
} // namespace

TEST(IndependentBorderSetTest, select_in_empty_graph) {
  auto graph = make_empty_graph();
  auto p_graph = make_partitioned_graph_by_rank(graph);

  auto is = find_independent_border_set(p_graph, 0);
  ASSERT_TRUE(is.empty());
}

TEST(IndependentBorderSetTest, select_in_circle_graph) {
  auto graph = make_circle_graph();
  auto p_graph = make_partitioned_graph_by_rank(graph);

  auto is = find_independent_border_set(p_graph, 0);

  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  if (size == 1) { // No border on just one PE
    ASSERT_TRUE(is.empty());
  } else {
    expect_nonempty_independent_set(p_graph, is);
  }
}

TEST(IndependentBorderSetTest, select_in_cut_edge_graph_10) {
  for (const NodeID num_nodes_per_pe : {1, 2, 8, 16, 32, 64}) {
    auto graph = make_cut_edge_graph(num_nodes_per_pe);
    auto p_graph = make_partitioned_graph_by_rank(graph);
    auto is = find_independent_border_set(p_graph, 0);
    auto global_is = allgather_independent_set(p_graph, is);

    const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
    if (size == 1) {
      ASSERT_TRUE(global_is.empty());
    } else {
      ASSERT_EQ(global_is.size(), size * num_nodes_per_pe);
      expect_nonempty_independent_set(p_graph, is);
    }
  }
}

TEST(IndependentBorderSetTest, randomization_works) {
  auto graph = make_cut_edge_graph(1000);
  auto p_graph = make_partitioned_graph_by_rank(graph);

  auto a_is = find_independent_border_set(p_graph, 0);
  auto a_global_is = allgather_independent_set(p_graph, a_is);

  auto b_is = find_independent_border_set(p_graph, 1);
  auto b_global_is = allgather_independent_set(p_graph, b_is);

  // prob. for same IS negetible
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  if (size == 1) {
    EXPECT_EQ(a_global_is, b_global_is);
  } else {
    EXPECT_NE(a_global_is, b_global_is);
  }
}
} // namespace kaminpar::dist::graph
