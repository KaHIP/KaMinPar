/*******************************************************************************
 * @file:   move_conflict_resolver_test.cc
 * @author: Daniel Seemaier
 * @date:   23.08.2022
 * @brief:  Unit tests for global move conflict resolution.
 ******************************************************************************/
#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-mpi/utils.h"

#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/refinement/fm/move_conflict_resolver.h"

namespace kaminpar::dist {
namespace {
GlobalMove create_dummy_move(const GlobalNodeID node, const NodeID group, const EdgeWeight gain) {
  return {node, group, 0, gain, 0, 0};
}
} // namespace

TEST(GlobalMoveConflictResolver, empty_move_set) {
  std::vector<GlobalMove> my_global_moves;

  auto resolved_moved = broadcast_and_resolve_global_moves(my_global_moves, MPI_COMM_WORLD);
  EXPECT_TRUE(resolved_moved.empty());
}

TEST(GlobalMoveConflictResolver, single_node_groups_with_rank_gain) {
  std::vector<GlobalMove> my_global_moves;
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  // Move the same two nodes on all PEs
  my_global_moves.push_back(create_dummy_move(0, 0, rank));
  my_global_moves.push_back(create_dummy_move(1, 1, rank));

  auto resolved = broadcast_and_resolve_global_moves(my_global_moves, MPI_COMM_WORLD);

  // The winner should be the moves by the PE with the highest rank
  ASSERT_EQ(resolved.size(), 2 * size);
  for (const auto &move : resolved) {
    if (move.gain != size - 1) {
      EXPECT_EQ(move.node, kInvalidGlobalNodeID);
    } else {
      EXPECT_THAT(move.node, ::testing::AnyOf(0, 1));
    }
  }
}
} // namespace kaminpar::dist
