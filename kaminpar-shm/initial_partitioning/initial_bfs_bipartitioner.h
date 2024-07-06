/*******************************************************************************
 * BFS based initial bipartitioner.
 *
 * @file:   initial_bfs_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <array>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/initial_partitioning/initial_flat_bipartitioner.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/queue.h"

namespace kaminpar::shm {
namespace bfs {
struct alternating;   // Switch between queues after each node
struct lighter;       // Use lighter queue next
struct sequential;    // Only use the first queue
struct longer_queue;  // Use longer queue next
struct shorter_queue; // Use shorter queue next
} // namespace bfs

/*!
 * Grows two blocks starting from pseudo peripheral nodes using a breath first
 * search.
 *
 * In each step, a node is assigned to one of the two blocks. A block selection
 * strategy is used to decide when to switch between blocks.
 *
 * @tparam BlockSelectionStrategy Invoked after each step to choose the active block.
 */
template <typename BlockSelectionStrategy>
class InitialBFSBipartitioner : public InitialFlatBipartitioner {
  static constexpr std::size_t kMarkAssigned = 2;

public:
  InitialBFSBipartitioner(const InitialPoolPartitionerContext &pool_ctx);

  void init(const CSRGraph &graph, const PartitionContext &p_ctx) final;

protected:
  void fill_bipartition() final;

private:
  Marker<> _bfs_init_marker{};
  const int _num_seed_iterations;

  std::array<Queue<NodeID>, 2> _queues{};
  Marker<3> _marker{};
};

using AlternatingBfsBipartitioner = InitialBFSBipartitioner<bfs::alternating>;
using LighterBlockBfsBipartitioner = InitialBFSBipartitioner<bfs::lighter>;
using SequentialBfsBipartitioner = InitialBFSBipartitioner<bfs::sequential>;
using LongerQueueBfsBipartitioner = InitialBFSBipartitioner<bfs::longer_queue>;
using ShorterQueueBfsBipartitioner = InitialBFSBipartitioner<bfs::shorter_queue>;
} // namespace kaminpar::shm
