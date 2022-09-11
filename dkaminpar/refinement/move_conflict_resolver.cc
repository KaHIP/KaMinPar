/*******************************************************************************
 * @file:   move_conflict_resolver.cc
 * @author: Daniel Seemaier
 * @date:   11.09.2022
 * @brief:  Resolve conflicts between move sequences by accepting moves from
 * the best group.
 ******************************************************************************/
#include "dkaminpar/refinement/move_conflict_resolver.h"

#include <unordered_set>

#include "dkaminpar/mpi/wrapper.h"

#include "common/asserting_cast.h"
#include "common/logger.h"
#include "common/parallel/algorithm.h"
#include "common/timer.h"

namespace kaminpar::dist {
namespace {
SET_DEBUG(false);
}

std::vector<GlobalMove> allgather_global_moves(std::vector<GlobalMove>& my_global_moves, MPI_Comm comm) {
    SCOPED_TIMER("Allgather global moves");

    // Make groups unique by PE
    const PEID rank = mpi::get_comm_rank(comm);
    for (auto& move: my_global_moves) {
        move.group = move.group << 32 | rank;
    }

    // Allgather number of global moves on each PE
    const int  my_moves_count = asserting_cast<int>(my_global_moves.size());
    const auto recv_counts    = mpi::allgather(my_moves_count, comm);

    std::vector<int> recv_displs(recv_counts.size() + 1);
    parallel::prefix_sum(recv_counts.begin(), recv_counts.end(), recv_displs.begin() + 1);

    // Allgather global moves
    std::vector<GlobalMove> result(recv_displs.back());
    mpi::allgatherv<GlobalMove, GlobalMove>(
        my_global_moves.data(), my_moves_count, result.data(), recv_counts.data(), recv_displs.data(), comm
    );
    return result;
}

void sort_and_compress_move_groups(std::vector<GlobalMove>& global_moves) {
    SCOPED_TIMER("Sort and compress move groups");

    // Sort by move group
    std::sort(global_moves.begin(), global_moves.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.group < rhs.group;
    });

    // Compute group gains
    std::vector<EdgeWeight> group_gains;
    for (std::size_t i = 0; i < global_moves.size();) {
        std::int64_t current_group = global_moves[i].group;
        EdgeWeight group_gain      = 0;

        while (i < global_moves.size() && global_moves[i].group == current_group) {
            group_gain += global_moves[i].gain;
            global_moves[i].group = group_gains.size();
            ++i;
        }

        group_gains.emplace_back(group_gain);
    }

    // Sort by group gains
    std::sort(global_moves.begin(), global_moves.end(), [&group_gains](const auto& lhs, const auto& rhs) {
        return group_gains[lhs.group] > group_gains[rhs.group];
    });
}

void resolve_move_conflicts_greedy(std::vector<GlobalMove>& global_moves) {
    SCOPED_TIMER("Resolve move conflicts");

    std::unordered_set<GlobalNodeID> moved_nodes;

    for (std::size_t i = 0; i < global_moves.size();) {
        std::int64_t current_group  = global_moves[i].group;
        bool         found_conflict = false;

        // First pass: check that no node has been moved before
        for (std::size_t j = i; j < global_moves.size() && global_moves[j].group == current_group && !found_conflict;
             ++j) {
            const GlobalNodeID current_node = global_moves[j].node;
            found_conflict                  = moved_nodes.find(current_node) != moved_nodes.end();
        }

        // Second pass: mark all moves in this group as invalid or add them to the moved nodes
        for (std::size_t j = i; j < global_moves.size() && global_moves[j].group == current_group; ++j) {
            if (found_conflict) {
                DBG << "Lost: " << V(global_moves[j].node) << V(global_moves[j].gain);
                global_moves[j].node = kInvalidGlobalNodeID; // mark move as do not take
            } else {
                const GlobalNodeID current_node = global_moves[j].node;
                moved_nodes.insert(current_node);
            }

            ++i;
        }
    }
}

std::vector<GlobalMove> broadcast_and_resolve_global_moves(std::vector<GlobalMove>& my_global_moves, MPI_Comm comm) {
    auto global_moves = allgather_global_moves(my_global_moves, comm);
    sort_and_compress_move_groups(global_moves);
    resolve_move_conflicts_greedy(global_moves);
    return global_moves;
}
} // namespace kaminpar::dist
