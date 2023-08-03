/*******************************************************************************
 * Resolve conflicts between move sequences by accepting moves from the best
 * group.
 *
 * @file:   move_conflict_resolver.cc
 * @author: Daniel Seemaier
 * @date:   11.09.2022
 ******************************************************************************/
#include "dkaminpar/refinement/fm/move_conflict_resolver.h"

#include <unordered_set>

#include "dkaminpar/dkaminpar.h"
#include "dkaminpar/mpi/wrapper.h"

#include "common/asserting_cast.h"
#include "common/logger.h"
#include "common/parallel/algorithm.h"
#include "common/timer.h"

namespace kaminpar::dist {
namespace {
SET_DEBUG(false);
}

std::vector<GlobalMove>
allgather_global_moves(std::vector<GlobalMove> &my_global_moves, MPI_Comm comm) {
  SCOPED_TIMER("Allgather global moves");

  DBG << "Contributing " << my_global_moves.size() << " moves";

  // Make groups unique by PE
  START_TIMER("Make groups unique");
  const PEID rank = mpi::get_comm_rank(comm);
  for (auto &move : my_global_moves) {
    move.group = move.group << 32 | rank;
  }
  STOP_TIMER();

  // Allgather number of global moves on each PE
  START_TIMER("Allgather move counts");
  const int my_moves_count = asserting_cast<int>(my_global_moves.size());
  const auto recv_counts = mpi::allgather(my_moves_count, comm);
  STOP_TIMER();

  START_TIMER("Compute move displs");
  std::vector<int> recv_displs(recv_counts.size() + 1);
  parallel::prefix_sum(recv_counts.begin(), recv_counts.end(), recv_displs.begin() + 1);
  STOP_TIMER();

  // Allgather global moves
  START_TIMER("Allgathering moves");
  std::vector<GlobalMove> result(recv_displs.back());
  mpi::allgatherv<GlobalMove, GlobalMove>(
      my_global_moves.data(),
      my_moves_count,
      result.data(),
      recv_counts.data(),
      recv_displs.data(),
      comm
  );
  STOP_TIMER();

  return result;
}

void sort_move_groups(std::vector<GlobalMove> &global_moves) {
  SCOPED_TIMER("Sort move groups");
  std::sort(global_moves.begin(), global_moves.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.gain > rhs.gain || (lhs.gain == rhs.gain && lhs.group < rhs.group);
  });
}

void sort_and_compress_move_groups(std::vector<GlobalMove> &global_moves) {
  SCOPED_TIMER("Sort and compress move groups");

  // Sort by move group
  std::sort(global_moves.begin(), global_moves.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.group < rhs.group;
  });

  // Compute group gains
  std::vector<EdgeWeight> group_gains;
  for (std::size_t i = 0; i < global_moves.size();) {
    std::int64_t current_group = global_moves[i].group;
    EdgeWeight group_gain = 0;

    while (i < global_moves.size() && global_moves[i].group == current_group) {
      group_gain += global_moves[i].gain;
      global_moves[i].group = group_gains.size();
      ++i;
    }

    group_gains.emplace_back(group_gain);
  }

  // Sort by group gains
  std::sort(
      global_moves.begin(),
      global_moves.end(),
      [&group_gains](const auto &lhs, const auto &rhs) {
        return group_gains[lhs.group] > group_gains[rhs.group];
      }
  );
}

void resolve_move_conflicts_greedy(std::vector<GlobalMove> &global_moves) {
  SCOPED_TIMER("Resolve move conflicts");

  std::unordered_set<GlobalNodeID> moved_nodes;

  for (std::size_t i = 0; i < global_moves.size();) {
    std::int64_t current_group = global_moves[i].group;
    bool found_conflict = false;

    // First pass: check that no node has been moved before
    for (std::size_t j = i;
         j < global_moves.size() && global_moves[j].group == current_group && !found_conflict;
         ++j) {
      const GlobalNodeID current_node = global_moves[j].node;
      found_conflict = moved_nodes.find(current_node) != moved_nodes.end();
    }

    // Second pass: mark all moves in this group as invalid or add them to the
    // moved nodes
    for (std::size_t j = i; j < global_moves.size() && global_moves[j].group == current_group;
         ++j) {
      if (found_conflict) {
        global_moves[j].node = invalidate_id(global_moves[j].node);
      } else {
        const GlobalNodeID current_node = global_moves[j].node;
        moved_nodes.insert(current_node);
      }

      ++i;
    }
  }
}

std::vector<GlobalMove>
broadcast_and_resolve_global_moves(std::vector<GlobalMove> &my_global_moves, MPI_Comm comm) {
  DBG << "Got " << my_global_moves.size() << " global moves on this PE";

  // Resolve conflicts locally
  START_TIMER("Local conflict resolution");
  sort_move_groups(my_global_moves);
  resolve_move_conflicts_greedy(my_global_moves);
  STOP_TIMER();

  // Filter
  std::vector<GlobalMove> my_filtered_global_moves;
  for (const auto &move : my_global_moves) {
    if (is_valid_id(move.node)) {
      my_filtered_global_moves.push_back(move);
    }
  }

  DBG << "After resolving local conflicts: " << my_filtered_global_moves.size()
      << " global moves on this PE";

  auto global_moves = allgather_global_moves(my_filtered_global_moves, comm);

  DBG << "After allgathering: " << global_moves.size() << " global moves";

  START_TIMER("Global conflict resolution");
  sort_move_groups(global_moves);
  resolve_move_conflicts_greedy(global_moves);
  STOP_TIMER();

  return global_moves;
}
} // namespace kaminpar::dist
