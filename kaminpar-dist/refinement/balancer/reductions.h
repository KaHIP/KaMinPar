/*******************************************************************************
 * Candidate reductions for the greedy balancers.
 *
 * @file:   reductions.h
 * @author: Daniel Seemaier
 * @date:   27.08.2023
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-mpi/binary_reduction_tree.h"

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/refinement/balancer/weight_buckets.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::dist {
template <typename Candidate>
std::vector<Candidate> reduce_candidates(
    std::vector<Candidate> sendbuf,
    const NodeID max_nodes_per_block,
    const DistributedPartitionedGraph &p_graph,
    const PartitionContext &p_ctx
) {
  SCOPED_TIMER("Reduce candidates");

  auto kInvalidID = [&] {
    if constexpr (std::is_same_v<decltype(Candidate::id), GlobalNodeID>) {
      return kInvalidGlobalNodeID;
    } else {
      return kInvalidNodeID;
    }
  }();

  return mpi::perform_binary_reduction(
      std::move(sendbuf),
      std::vector<Candidate>{},
      [&](std::vector<Candidate> lhs, std::vector<Candidate> rhs) {
        // Precondition: candidates must be sorted by their from blocks
        auto check_sorted_by_from = [](const auto &candidates) {
          for (std::size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].from < candidates[i - 1].from) {
              return false;
            }
          }
          return true;
        };
        KASSERT(
            check_sorted_by_from(rhs) && check_sorted_by_from(lhs),
            "rhs or lhs candidates are not sorted by their .from property"
        );

        std::size_t idx_lhs = 0;
        std::size_t idx_rhs = 0;
        std::vector<BlockWeight> block_weight_deltas(p_graph.k());
        std::vector<Candidate> winners;

        while (idx_lhs < lhs.size() && idx_rhs < rhs.size()) {
          const BlockID from = std::min(lhs[idx_lhs].from, rhs[idx_rhs].from);

          // Find regions in `rhs` and `lhs` with move sets in block `from`
          std::size_t idx_lhs_end = idx_lhs;
          std::size_t idx_rhs_end = idx_rhs;
          while (idx_lhs_end < lhs.size() && lhs[idx_lhs_end].from == from) {
            ++idx_lhs_end;
          }
          while (idx_rhs_end < rhs.size() && rhs[idx_rhs_end].from == from) {
            ++idx_rhs_end;
          }

          // Merge regions
          const std::size_t lhs_count = idx_lhs_end - idx_lhs;
          const std::size_t rhs_count = idx_rhs_end - idx_rhs;
          const std::size_t count = lhs_count + rhs_count;

          std::vector<Candidate> candidates(count);
          std::copy(lhs.begin() + idx_lhs, lhs.begin() + idx_lhs_end, candidates.begin());
          std::copy(
              rhs.begin() + idx_rhs, rhs.begin() + idx_rhs_end, candidates.begin() + lhs_count
          );
          std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
            return a.gain > b.gain;
          });

          // Pick feasible prefix
          NodeWeight total_weight = 0;
          std::size_t num_rejected_candidates = 0;
          std::size_t num_accepted_candidates = 0;

          for (std::size_t i = 0; i < count; ++i) {
            const BlockID to = candidates[i].to;
            const NodeWeight weight = candidates[i].weight;

            // Reject the move set candidate if it would overload the target block
            if (from != to && p_graph.block_weight(to) + block_weight_deltas[to] + weight >
                                  p_ctx.graph->max_block_weight(to)) {
              candidates[i].id = kInvalidID;
              ++num_rejected_candidates;
            } else {
              block_weight_deltas[to] += weight;
              total_weight += weight;
              ++num_accepted_candidates;

              const BlockWeight overload =
                  p_graph.block_weight(from) - p_ctx.graph->max_block_weight(from);
              if (total_weight >= overload || num_accepted_candidates >= max_nodes_per_block) {
                break;
              }
            }
          }

          // Remove rejected candidates
          for (std::size_t i = 0; i < num_accepted_candidates; ++i) {
            while (candidates[i].id == kInvalidID) {
              std::swap(
                  candidates[i], candidates[num_accepted_candidates + num_rejected_candidates - 1]
              );
              --num_rejected_candidates;
            }
          }

          winners.insert(
              winners.end(), candidates.begin(), candidates.begin() + num_accepted_candidates
          );

          idx_lhs = idx_lhs_end;
          idx_rhs = idx_rhs_end;
        }

        // Keep remaining nodes
        auto add_remaining_candidates = [&](const auto &vec, std::size_t i) {
          for (; i < vec.size(); ++i) {
            const BlockID from = vec[i].from;
            const BlockID to = vec[i].to;
            const NodeWeight weight = vec[i].weight;

            if (from == to || p_graph.block_weight(to) + block_weight_deltas[to] + weight <=
                                  p_ctx.graph->max_block_weight(to)) {
              winners.push_back(vec[i]);
              if (from != to) {
                block_weight_deltas[to] += weight;
              }
            }
          }
        };

        add_remaining_candidates(lhs, idx_lhs);
        add_remaining_candidates(rhs, idx_rhs);

        return winners;
      },
      p_graph.communicator()
  );
}

inline StaticArray<GlobalNodeWeight>
reduce_buckets_binary_tree(const Buckets &buckets, const DistributedPartitionedGraph &p_graph) {
  SCOPED_TIMER("Reduce buckets");

  auto compactified = buckets.compactify();
  StaticArray<GlobalNodeWeight> empty(compactified.size());

  return mpi::perform_binary_reduction(
      std::move(compactified),
      std::move(empty),
      [&](auto lhs, auto rhs) {
        for (std::size_t i = 0; i < lhs.size(); ++i) {
          lhs[i] += rhs[i];
        }
        return std::move(lhs);
      },
      p_graph.communicator()
  );
}

inline StaticArray<GlobalNodeWeight>
reduce_buckets_mpireduce(const Buckets &buckets, const DistributedPartitionedGraph &p_graph) {
  SCOPED_TIMER("Reduce buckets");

  const PEID rank = mpi::get_comm_rank(p_graph.communicator());
  auto compactified = buckets.compactify();
  if (rank == 0) {
    MPI_Reduce(
        MPI_IN_PLACE,
        compactified.data(),
        compactified.size(),
        mpi::type::get<GlobalNodeWeight>(),
        MPI_SUM,
        0,
        p_graph.communicator()
    );
  } else {
    MPI_Reduce(
        compactified.data(),
        nullptr,
        compactified.size(),
        mpi::type::get<GlobalNodeWeight>(),
        MPI_SUM,
        0,
        p_graph.communicator()
    );
  }
  return compactified;
}
} // namespace kaminpar::dist
