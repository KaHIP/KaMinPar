/*******************************************************************************
 * Resolve conflicts between move sequences by accepting moves from the best
 * group.
 *
 * @file:   move_conflict_resolver.h
 * @author: Daniel Seemaier
 * @date:   11.09.2022
 ******************************************************************************/
#pragma once

#include <utility>
#include <vector>

#include <mpi.h>

#include "dkaminpar/dkaminpar.h"

namespace kaminpar::dist {
struct GlobalMove {
  GlobalNodeID node;
  GlobalNodeID group;
  NodeWeight weight;
  EdgeWeight gain;
  BlockID from;
  BlockID to;
};

inline GlobalNodeID invalidate_id(const GlobalNodeID node) {
  return node | (1ul << 63);
}

inline bool is_invalid_id(const GlobalNodeID node) {
  return node & (1ul << 63);
}

inline bool is_valid_id(const GlobalNodeID node) {
  return !is_invalid_id(node);
}

inline GlobalNodeID extract_id(const GlobalNodeID node) {
  return is_invalid_id(node) ? node & ~(1ul << 63) : node;
}

std::vector<GlobalMove>
allgather_global_moves(std::vector<GlobalMove> &my_global_moves, MPI_Comm comm);

void sort_and_compress_move_groups(std::vector<GlobalMove> &global_moves);

void resolve_move_conflicts_greedy(std::vector<GlobalMove> &global_moves);

std::vector<GlobalMove>
broadcast_and_resolve_global_moves(std::vector<GlobalMove> &my_global_moves, MPI_Comm comm);
} // namespace kaminpar::dist
