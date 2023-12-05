/*******************************************************************************
 * Distributes block-induced subgraphs of a partitioned graph across PEs.
 *
 * @file:   graph_extraction.cc
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 ******************************************************************************/
#include "kaminpar-dist/graphutils/subgraph_extractor.h"

#include <algorithm>
#include <functional>

#include <mpi.h>

#include "kaminpar-mpi/alltoall.h"
#include "kaminpar-mpi/sparse_alltoall.h"
#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/graphutils/communication.h"
#include "kaminpar-dist/graphutils/synchronization.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/vector_ets.h"

namespace kaminpar::dist::graph {
SET_DEBUG(false);

namespace {
PEID compute_block_owner(const BlockID b, const BlockID k, const PEID num_pes) {
  return static_cast<PEID>(
      math::compute_local_range_rank<BlockID>(k, static_cast<BlockID>(num_pes), b)
  );
}

auto count_block_induced_subgraph_sizes(const DistributedPartitionedGraph &p_graph) {
  parallel::vector_ets<NodeID> num_nodes_per_block_ets(p_graph.k());
  parallel::vector_ets<EdgeID> num_edges_per_block_ets(p_graph.k());

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, p_graph.n()), [&](const auto r) {
    auto &num_nodes_per_block = num_nodes_per_block_ets.local();
    auto &num_edges_per_block = num_edges_per_block_ets.local();
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      const BlockID u_block = p_graph.block(u);
      ++num_nodes_per_block[u_block];
      for (const auto [e, v] : p_graph.neighbors(u)) {
        if (u_block == p_graph.block(v)) {
          ++num_edges_per_block[u_block];
        }
      }
    }
  });

  return std::make_pair(
      num_nodes_per_block_ets.combine(std::plus{}), num_edges_per_block_ets.combine(std::plus{})
  );
}
} // namespace

// Build a local block-induced subgraph for each block of the graph partition.
ExtractedLocalSubgraphs
extract_local_block_induced_subgraphs(const DistributedPartitionedGraph &p_graph) {
  mpi::barrier(p_graph.communicator());
  SCOPED_TIMER("Extracting local block induced subgraphs");

  auto [num_nodes_per_block, num_edges_per_block] = count_block_induced_subgraph_sizes(p_graph);
  const EdgeID num_internal_edges =
      std::accumulate(num_edges_per_block.begin(), num_edges_per_block.end(), 0);

  ExtractedLocalSubgraphs memory;
  auto &shared_nodes = memory.shared_nodes;
  auto &shared_node_weights = memory.shared_node_weights;
  auto &shared_edges = memory.shared_edges;
  auto &shared_edge_weights = memory.shared_edge_weights;
  auto &nodes_offset = memory.nodes_offset;
  auto &edges_offset = memory.edges_offset;
  auto &mapping = memory.mapping;
  auto next_node_in_subgraph = std::vector<parallel::Atomic<NodeID>>();

  // Allocate memory @todo
  {
    SCOPED_TIMER("Allocation");

    const std::size_t min_nodes_size = p_graph.n();
    const std::size_t min_edges_size = num_internal_edges;
    const std::size_t min_offset_size = p_graph.k() + 1;
    const std::size_t min_mapping_size = p_graph.total_n();

    KASSERT(shared_nodes.size() == shared_node_weights.size());
    KASSERT(shared_edges.size() == shared_edge_weights.size());
    KASSERT(nodes_offset.size() == edges_offset.size());

    if (shared_nodes.size() < min_nodes_size) {
      shared_nodes.resize(min_nodes_size);
      shared_node_weights.resize(min_nodes_size);
    }
    if (shared_edges.size() < min_edges_size) {
      shared_edges.resize(min_edges_size);
      shared_edge_weights.resize(min_edges_size);
    }
    if (nodes_offset.size() < min_offset_size) {
      nodes_offset.resize(min_offset_size);
      edges_offset.resize(min_offset_size);
    }
    if (mapping.size() < min_mapping_size) {
      mapping.resize(min_mapping_size);
    }

    next_node_in_subgraph.resize(p_graph.k());

    mpi::barrier(p_graph.communicator());
  }

  // Compute of graphs in shared_* arrays
  {
    SCOPED_TIMER("Compute subgraph offsets");

    parallel::prefix_sum(
        num_nodes_per_block.begin(), num_nodes_per_block.end(), nodes_offset.begin() + 1
    );
    parallel::prefix_sum(
        num_edges_per_block.begin(), num_edges_per_block.end(), edges_offset.begin() + 1
    );

    mpi::barrier(p_graph.communicator());
  }

  // Compute node ID offset of local subgraph in global subgraphs
  START_TIMER("Compute offsets");
  std::vector<NodeID> global_node_offset(p_graph.k());
  mpi::exscan(
      num_nodes_per_block.data(),
      global_node_offset.data(),
      p_graph.k(),
      MPI_SUM,
      p_graph.communicator()
  );
  mpi::barrier(p_graph.communicator());
  STOP_TIMER();

  // Build mapping from node IDs in p_graph to node IDs in the extracted
  // subgraph
  {
    SCOPED_TIMER("Build node mapping");

    // @todo bottleneck for scalability
    p_graph.pfor_nodes([&](const NodeID u) {
      const BlockID b = p_graph.block(u);
      const NodeID pos_in_subgraph = next_node_in_subgraph[b]++;
      const NodeID pos = nodes_offset[b] + pos_in_subgraph;
      shared_nodes[pos] = u;
      mapping[u] = global_node_offset[b] + pos_in_subgraph;
    });

    mpi::barrier(p_graph.communicator());
  }

  // Build mapping from local extract subgraph to global extracted subgraph for
  // ghost nodes
  START_TIMER("Node mapping allocation");
  std::vector<NodeID> global_ghost_node_mapping(p_graph.ghost_n());
  mpi::barrier(p_graph.communicator());
  STOP_TIMER();

  {
    SCOPED_TIMER("Exchange ghost node mapping");

    struct NodeToMappedNode {
      GlobalNodeID global_node;
      NodeID mapped_node;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<NodeToMappedNode>(
        p_graph.graph(),
        [&](const NodeID u) {
          return NodeToMappedNode{p_graph.local_to_global_node(u), mapping[u]};
        },
        [&](const auto buffer) {
          tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
            const auto &[global_node, mapped_node] = buffer[i];
            const NodeID local_node = p_graph.global_to_local_node(global_node);
            mapping[local_node] = mapped_node;
          });
        }
    );
    mpi::barrier(p_graph.communicator());
  }

  // Extract the subgraphs
  {
    SCOPED_TIMER("Extract subgraphs");

    tbb::parallel_for<BlockID>(0, p_graph.k(), [&](const BlockID b) {
      const NodeID n0 = nodes_offset[b];
      const EdgeID e0 = edges_offset[b];
      EdgeID e = 0;

      // u, v, e = IDs in extracted subgraph
      // u_prime, v_prime, e_prime = IDs in p_graph
      for (NodeID u = 0; u < next_node_in_subgraph[b]; ++u) {
        const NodeID pos = n0 + u;
        const NodeID u_prime = shared_nodes[pos];

        for (const auto [e_prime, v_prime] : p_graph.neighbors(u_prime)) {
          if (p_graph.block(v_prime) != b) {
            continue;
          }

          shared_edge_weights[e0 + e] = p_graph.edge_weight(e_prime);
          shared_edges[e0 + e] = mapping[v_prime];
          ++e;
        }

        shared_nodes[pos] = e;
        shared_node_weights[pos] = p_graph.node_weight(u_prime);
      }
    });
  }

  // remove global node offset in mapping -- we need the PE-relative value when
  // copying the subgraph partitions pack to the original graph
  p_graph.pfor_nodes([&](const NodeID u) {
    const BlockID b = p_graph.block(u);
    mapping[u] -= global_node_offset[b];
  });

  return memory;
}

BlockExtractionOffsets::BlockExtractionOffsets(PEID size, BlockID k)
    : _size(size),
      _k(k),
      _min_blocks_per_pe(k / size),
      _rem_blocks(k % size),
      _min_pes_per_block(size / k),
      _rem_pes(size % k) {}

BlockID BlockExtractionOffsets::first_block_on_pe(PEID pe) const {
  if (_size <= _k) {
    return pe * _min_blocks_per_pe + std::min<BlockID>(pe, _rem_blocks);
  } else {
    if (pe <= _rem_pes * _min_pes_per_block) {
      return pe / (_min_pes_per_block + 1);
    } else {
      return _rem_pes + (pe - _rem_pes * (_min_pes_per_block + 1)) / _min_pes_per_block;
    }
  }
}
BlockID BlockExtractionOffsets::first_invalid_block_on_pe(PEID pe) const {
  return first_block_on_pe(pe) + num_blocks_on_pe(pe);
}

BlockID BlockExtractionOffsets::num_blocks_on_pe(PEID pe) const {
  return std::max<BlockID>(1, _min_blocks_per_pe + (pe < _rem_blocks));
}

PEID BlockExtractionOffsets::first_pe_with_block(BlockID block) const {
  return std::max<BlockID>(block, block * _min_pes_per_block + std::min<BlockID>(block, _rem_pes));
}

PEID BlockExtractionOffsets::first_invalid_pe_with_block(BlockID block) const {
  return first_pe_with_block(block) + num_pes_with_block(block);
}

PEID BlockExtractionOffsets::num_pes_with_block(BlockID block) const {
  return std::max<PEID>(1, _min_pes_per_block + (block < _rem_pes));
}

namespace {
struct GraphSize {
  NodeID n;
  EdgeID m;

  GraphSize operator+(const GraphSize other) {
    return {n + other.n, m + other.m};
  }

  GraphSize &operator+=(const GraphSize other) {
    n += other.n;
    m += other.m;
    return *this;
  }
};

// Let B := number of blocks assigned to this PE
// Returns a vector vec of size B * P s.t.
//   vec[p * B + b] = size of the b-th block assigned to this PE coming from PE p
NoinitVector<GraphSize> exchange_subgraph_sizes(
    const DistributedPartitionedGraph &p_graph,
    const ExtractedLocalSubgraphs &memory,
    const BlockExtractionOffsets &offsets
) {
  SCOPED_TIMER("Exchange subgraph sizes");

  const PEID size = mpi::get_comm_size(p_graph.communicator());
  const PEID rank = mpi::get_comm_rank(p_graph.communicator());

  const std::size_t sendbuf_size = std::max<std::size_t>(p_graph.k(), size);
  NoinitVector<GraphSize> send_subgraph_sizes(sendbuf_size);

  p_graph.pfor_blocks([&](const BlockID block) {
    const PEID first_pe = offsets.first_pe_with_block(block);
    const PEID first_invalid_pe = offsets.first_invalid_pe_with_block(block);
    for (PEID to = first_pe; to < first_invalid_pe; ++to) {
      send_subgraph_sizes[to].n = memory.nodes_offset[block + 1] - memory.nodes_offset[block];
      send_subgraph_sizes[to].m = memory.edges_offset[block + 1] - memory.edges_offset[block];
    }
  });

  std::vector<int> sendcounts(size);
  std::vector<int> recvcounts(size);
  std::vector<int> sdispls(size);
  std::vector<int> rdispls(size);

  for (PEID pe = 0; pe < size; ++pe) {
    sendcounts[pe] = offsets.num_blocks_on_pe(pe);
  }
  std::fill(recvcounts.begin(), recvcounts.end(), offsets.num_blocks_on_pe(rank));
  std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), 0);
  std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);

  const std::size_t recvbuf_size = offsets.num_blocks_on_pe(rank) * size;
  NoinitVector<GraphSize> recv_subgraph_sizes(recvbuf_size);

  MPI_Alltoallv(
      send_subgraph_sizes.data(),
      sendcounts.data(),
      sdispls.data(),
      mpi::type::get<GraphSize>(),
      recv_subgraph_sizes.data(),
      recvcounts.data(),
      rdispls.data(),
      mpi::type::get<GraphSize>(),
      p_graph.communicator()
  );

  return recv_subgraph_sizes;
}

struct SharedGraphs {
  SharedGraphs(const NodeID n, const EdgeID m)
      : nodes(n),
        node_weights(n),
        edges(m),
        edge_weights(m) {}

  std::vector<EdgeID> nodes;
  std::vector<NodeWeight> node_weights;
  std::vector<NodeID> edges;
  std::vector<EdgeWeight> edge_weights;
};

SharedGraphs exchange_subgraphs(
    const DistributedPartitionedGraph &p_graph,
    ExtractedLocalSubgraphs &memory,
    const NoinitVector<GraphSize> &recv_subgraph_displs,
    const BlockExtractionOffsets &offsets
) {
  SCOPED_TIMER("Exchange subgraphs");
  const PEID size = mpi::get_comm_size(p_graph.communicator());
  const PEID rank = mpi::get_comm_rank(p_graph.communicator());

  std::vector<int> sendcounts_nodes(size);
  std::vector<int> sendcounts_edges(size);
  std::vector<int> sdispls_nodes(size + 1);
  std::vector<int> sdispls_edges(size + 1);
  std::vector<int> recvcounts_nodes(size);
  std::vector<int> recvcounts_edges(size);
  std::vector<int> rdispls_nodes(size + 1);
  std::vector<int> rdispls_edges(size + 1);

  START_TIMER("Compute counts and displs");
  tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
    // Compute sendcounts + sdispls
    {
      const BlockID first_block_on_pe = offsets.first_block_on_pe(pe);
      const BlockID first_invalid_block_on_pe = offsets.first_invalid_block_on_pe(pe);
      sendcounts_nodes[pe] =
          memory.nodes_offset[first_invalid_block_on_pe] - memory.nodes_offset[first_block_on_pe];
      sendcounts_edges[pe] =
          memory.edges_offset[first_invalid_block_on_pe] - memory.edges_offset[first_block_on_pe];

      const BlockID displs = offsets.first_block_on_pe(pe + 1);
      sdispls_nodes[pe + 1] = memory.nodes_offset[displs];
      sdispls_edges[pe + 1] = memory.edges_offset[displs];
    }

    // Compute recvcounts + rdispls
    {
      auto compute_offset = [&](const BlockID pe) {
        const BlockID min_blocks_per_pe = p_graph.k() / size;
        const BlockID rem_blocks = p_graph.k() % size;
        return std::max<BlockID>(pe, pe * min_blocks_per_pe + std::min<BlockID>(pe, rem_blocks));
      };

      const BlockID first_block_on_pe = compute_offset(pe);
      const BlockID first_invalid_block_on_pe = compute_offset(pe + 1);

      recvcounts_nodes[pe] = recv_subgraph_displs[first_invalid_block_on_pe].n -
                             recv_subgraph_displs[first_block_on_pe].n;
      recvcounts_edges[pe] = recv_subgraph_displs[first_invalid_block_on_pe].m -
                             recv_subgraph_displs[first_block_on_pe].m;

      rdispls_nodes[pe + 1] = recv_subgraph_displs[first_invalid_block_on_pe].n;
      rdispls_edges[pe + 1] = recv_subgraph_displs[first_invalid_block_on_pe].m;
    }
  });
  MPI_Barrier(p_graph.communicator());
  STOP_TIMER();

  START_TIMER("Allocation");
  SharedGraphs shared_graphs(rdispls_nodes.back(), rdispls_edges.back());
  MPI_Barrier(p_graph.communicator());
  STOP_TIMER();

  START_TIMER("MPI_Alltoallv");
  mpi::sparse_alltoallv(
      memory.shared_nodes.data(),
      sendcounts_nodes.data(),
      sdispls_nodes.data(),
      shared_graphs.nodes.data(),
      recvcounts_nodes.data(),
      rdispls_nodes.data(),
      p_graph.communicator()
  );
  mpi::sparse_alltoallv(
      memory.shared_node_weights.data(),
      sendcounts_nodes.data(),
      sdispls_nodes.data(),
      shared_graphs.node_weights.data(),
      recvcounts_nodes.data(),
      rdispls_nodes.data(),
      p_graph.communicator()
  );
  mpi::sparse_alltoallv(
      memory.shared_edges.data(),
      sendcounts_edges.data(),
      sdispls_edges.data(),
      shared_graphs.edges.data(),
      recvcounts_edges.data(),
      rdispls_edges.data(),
      p_graph.communicator()
  );
  mpi::sparse_alltoallv(
      memory.shared_edge_weights.data(),
      sendcounts_edges.data(),
      sdispls_edges.data(),
      shared_graphs.edge_weights.data(),
      recvcounts_edges.data(),
      rdispls_edges.data(),
      p_graph.communicator()
  );
  STOP_TIMER();

  return shared_graphs;
}

std::pair<std::vector<shm::Graph>, std::vector<std::vector<NodeID>>> construct_subgraphs(
    const DistributedPartitionedGraph &p_graph,
    const SharedGraphs &shared_graphs,
    const NoinitVector<GraphSize> &recv_subgraph_sizes,
    const NoinitVector<GraphSize> &recv_subgraph_displs,
    const BlockExtractionOffsets &offsets
) {
  SCOPED_TIMER("Construct subgraphs");

  const PEID size = mpi::get_comm_size(p_graph.communicator());
  const PEID rank = mpi::get_comm_rank(p_graph.communicator());

  const auto &shared_nodes = shared_graphs.nodes;
  const auto &shared_node_weights = shared_graphs.node_weights;
  const auto &shared_edges = shared_graphs.edges;
  const auto &shared_edge_weights = shared_graphs.edge_weights;

  std::vector<shm::Graph> subgraphs(offsets.num_blocks_on_pe(rank));
  std::vector<std::vector<NodeID>> subgraphs_offsets(subgraphs.size());

  tbb::parallel_for<BlockID>(0, offsets.num_blocks_on_pe(rank), [&](const BlockID b) {
    NodeID n = 0;
    EdgeID m = 0;
    for (PEID pe = 0; pe < size; ++pe) {
      const std::size_t i = b + pe * offsets.num_blocks_on_pe(rank);
      n += recv_subgraph_sizes[i].n;
      m += recv_subgraph_sizes[i].m;
    }

    // Allocate memory for subgraph
    StaticArray<shm::EdgeID> subgraph_nodes(n + 1);
    StaticArray<shm::NodeWeight> subgraph_node_weights(n);
    StaticArray<shm::NodeID> subgraph_edges(m);
    StaticArray<shm::EdgeWeight> subgraph_edge_weights(m);

    // Copy subgraph to memory
    // @todo better approach might be to compute a prefix sum on
    // recv_subgraph_sizes
    NodeID pos_n = 0;
    EdgeID pos_m = 0;

    for (PEID pe = 0; pe < size; ++pe) {
      const std::size_t id = b + pe * offsets.num_blocks_on_pe(rank);

      const auto [num_nodes, num_edges] = recv_subgraph_sizes[id];
      const auto [offset_nodes, offset_edges] = recv_subgraph_displs[id];
      subgraphs_offsets[b].push_back(pos_n);

      std::copy(
          shared_nodes.begin() + offset_nodes,
          shared_nodes.begin() + offset_nodes + num_nodes,
          subgraph_nodes.begin() + pos_n + 1
      );
      std::copy(
          shared_node_weights.begin() + offset_nodes,
          shared_node_weights.begin() + offset_nodes + num_nodes,
          subgraph_node_weights.begin() + pos_n
      );
      std::copy(
          shared_edges.begin() + offset_edges,
          shared_edges.begin() + offset_edges + num_edges,
          subgraph_edges.begin() + pos_m
      );
      std::copy(
          shared_edge_weights.begin() + offset_edges,
          shared_edge_weights.begin() + offset_edges + num_edges,
          subgraph_edge_weights.begin() + pos_m
      );

      // Copied independent nodes arrays -- thus, offset segment by number of
      // edges received from previous PEs
      for (NodeID u = 0; u < num_nodes; ++u) {
        subgraph_nodes[pos_n + 1 + u] += pos_m;
      }

      pos_n += num_nodes;
      pos_m += num_edges;
    }
    subgraphs_offsets[b].push_back(pos_n);

    subgraphs[b] = shm::Graph(
        std::move(subgraph_nodes),
        std::move(subgraph_edges),
        std::move(subgraph_node_weights),
        std::move(subgraph_edge_weights),
        false
    );
  });

  return {std::move(subgraphs), std::move(subgraphs_offsets)};
}

std::pair<std::vector<shm::Graph>, std::vector<std::vector<NodeID>>> gather_block_induced_subgraphs(
    const DistributedPartitionedGraph &p_graph, ExtractedLocalSubgraphs &memory
) {
  SCOPED_TIMER("Gathering block induced subgraphs");

  const PEID size = mpi::get_comm_size(p_graph.communicator());
  const PEID rank = mpi::get_comm_rank(p_graph.communicator());
  BlockExtractionOffsets offsets(size, p_graph.k());

  IF_DBG0 {
    DBG0 << "PE summary:";
    for (PEID pe = 0; pe < size; ++pe) {
      DBG0 << "- PE " << pe << ": " << offsets.num_blocks_on_pe(pe)
           << " blocks: " << offsets.first_block_on_pe(pe) << " <= * < "
           << offsets.first_invalid_block_on_pe(pe);
    }
    DBG0 << "Block summary:";
    for (const BlockID block : p_graph.blocks()) {
      DBG0 << "- Block " << block << ": assigned to " << offsets.num_pes_with_block(block)
           << " PEs " << offsets.first_pe_with_block(block)
           << " < * <= " << offsets.first_invalid_pe_with_block(block);
    }
  }

  NoinitVector<GraphSize> recv_subgraph_sizes = exchange_subgraph_sizes(p_graph, memory, offsets);
  NoinitVector<GraphSize> recv_subgraph_displs(recv_subgraph_sizes.size() + 1);
  recv_subgraph_displs.front() = {0, 0};
  parallel::prefix_sum(
      recv_subgraph_sizes.begin(), recv_subgraph_sizes.end(), recv_subgraph_displs.begin() + 1
  );

  const auto shared_graphs = exchange_subgraphs(p_graph, memory, recv_subgraph_displs, offsets);

  return construct_subgraphs(
      p_graph, shared_graphs, recv_subgraph_sizes, recv_subgraph_displs, offsets
  );
}
} // namespace

ExtractedSubgraphs
extract_and_scatter_block_induced_subgraphs(const DistributedPartitionedGraph &p_graph) {
  // Catch special case: if there are no blocks, return nothing
  // (This is only meaningful if the graph is empty.)
  if (p_graph.k() == 0) {
    return {};
  }

  auto extracted_local_subgraphs = extract_local_block_induced_subgraphs(p_graph);
  auto [gathered_subgraphs, offsets] =
      gather_block_induced_subgraphs(p_graph, extracted_local_subgraphs);

  return {
      std::move(gathered_subgraphs),
      std::move(offsets),
      std::move(extracted_local_subgraphs.mapping)};
}

DistributedPartitionedGraph copy_subgraph_partitions(
    DistributedPartitionedGraph p_graph,
    const std::vector<shm::PartitionedGraph> &p_subgraphs,
    ExtractedSubgraphs &extracted_subgraphs
) {
  KASSERT(!p_subgraphs.empty());

  SCOPED_TIMER("Projecting subgraph partitions");
  const PEID size = mpi::get_comm_size(p_graph.communicator());
  const BlockExtractionOffsets offsets(size, p_graph.k());

  // If there are more PEs than blocks, multiple PEs can hold a copy of the same block-induced
  // subgraph. In this case, we have to dedice which partition to use. Handle this as a special
  // case.
  if (static_cast<BlockID>(size) > p_graph.k()) {
    return copy_duplicated_subgraph_partitions(
        std::move(p_graph), p_subgraphs, extracted_subgraphs
    );
  }

  const auto &subgraph_offsets = extracted_subgraphs.subgraph_offsets;
  const auto &mapping = extracted_subgraphs.mapping;

  // Assume that all subgraph partitions have the same number of blocks
  // @todo
  const PEID k_multiplier = p_subgraphs.front().k();
  const PEID new_k = p_graph.k() * k_multiplier;

  // Send new block IDs to the right PE
  std::vector<std::vector<BlockID>> partition_sendbufs(size);
  for (BlockID block = 0; block < p_subgraphs.size(); ++block) {
    const auto &p_subgraph = p_subgraphs[block];
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
      const NodeID from = subgraph_offsets[block][pe];
      const NodeID to = subgraph_offsets[block][pe + 1];
      for (NodeID u = from; u < to; ++u) {
        partition_sendbufs[pe].push_back(p_subgraph.block(u));
      }
    });
  }

  const auto partition_recvbufs =
      mpi::sparse_alltoall_get<BlockID>(partition_sendbufs, p_graph.communicator());

  // To index partition_recvbufs, we need the number of nodes *on our PE* in
  // each block
  parallel::vector_ets<NodeID> block_sizes_ets(p_graph.k());
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, p_graph.n()), [&](const auto &r) {
    auto &block_sizes = block_sizes_ets.local();
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      ++block_sizes[p_graph.block(u)];
    }
  });
  const auto block_sizes = block_sizes_ets.combine(std::plus{});

  std::vector<NodeID> block_offsets(p_graph.k() + 1);
  parallel::prefix_sum(block_sizes.begin(), block_sizes.end(), block_offsets.begin() + 1);

  // Assign nodes in p_graph to new blocks
  auto partition = p_graph.take_partition(); // NOTE: do not use p_graph after this
  p_graph.pfor_nodes([&](const NodeID u) {
    const BlockID block = partition[u];
    const PEID owner = offsets.first_pe_with_block(block);
    const BlockID first_block_on_owner = offsets.first_block_on_pe(owner);
    const BlockID block_offset = block_offsets[block] - block_offsets[first_block_on_owner];
    const NodeID mapped_u = mapping[u]; // ID of u in its block-induced subgraph
    const BlockID new_b = block * k_multiplier + partition_recvbufs[owner][mapped_u + block_offset];
    partition[u] = new_b;
  });

  // Create partitioned graph with the new partition
  DistributedPartitionedGraph new_p_graph(&p_graph.graph(), new_k, std::move(partition));

  // Synchronize block assignment of ghost nodes
  synchronize_ghost_node_block_ids(new_p_graph);

  KASSERT(
      debug::validate_partition(new_p_graph),
      "graph partition in inconsistent state",
      assert::heavy
  );
  return new_p_graph;
}

DistributedPartitionedGraph copy_duplicated_subgraph_partitions(
    DistributedPartitionedGraph p_graph,
    const std::vector<shm::PartitionedGraph> &p_subgraphs,
    ExtractedSubgraphs &extracted_subgraphs
) {
  KASSERT(!p_subgraphs.empty());

  if (p_subgraphs.size() > 1) {
    return copy_subgraph_partitions(std::move(p_graph), p_subgraphs, extracted_subgraphs);
  }

  SCOPED_TIMER("Projecting subgraph partitions");
  const shm::PartitionedGraph &p_subgraph = p_subgraphs.front();
  const PEID size = mpi::get_comm_size(p_graph.communicator());
  const PEID rank = mpi::get_comm_rank(p_graph.communicator());

  const BlockExtractionOffsets offsets(size, p_graph.k());

  const auto &subgraph_offsets = extracted_subgraphs.subgraph_offsets;
  const auto &mapping = extracted_subgraphs.mapping;

  const PEID k_multiplier = p_subgraph.k();
  const PEID new_k = p_graph.k() * k_multiplier;

  // Let all PEs know all cuts
  const GlobalEdgeWeight my_cut = shm::metrics::edge_cut(p_subgraph);
  const auto cuts = mpi::allgather<GlobalEdgeWeight>(my_cut, p_graph.communicator());

  // Decide whether pick the partition computed by a certain PE
  auto find_winner = [&](const BlockID block) {
    PEID min_cut_pe = offsets.first_pe_with_block(block);
    for (PEID cur = min_cut_pe; cur < offsets.first_invalid_pe_with_block(block); ++cur) {
      if (cuts[cur] < cuts[min_cut_pe]) {
        min_cut_pe = cur;
      }
    }
    return min_cut_pe;
  };

  // Only send our partition if our partition is the best:
  std::vector<std::vector<BlockID>> partition_sendbufs(size);
  const BlockID our_block = offsets.first_block_on_pe(rank);
  if (find_winner(our_block) == rank) {
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
      const NodeID from = subgraph_offsets[0][pe];
      const NodeID to = subgraph_offsets[0][pe + 1];
      for (NodeID u = from; u < to; ++u) {
        partition_sendbufs[pe].push_back(p_subgraph.block(u));
      }
    });
  }
  const auto partition_recvbufs =
      mpi::sparse_alltoall_get<BlockID>(partition_sendbufs, p_graph.communicator());

  // To index partition_recvbufs, we need the number of nodes *on our PE* in
  // each block
  parallel::vector_ets<NodeID> block_sizes_ets(p_graph.k());
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, p_graph.n()), [&](const auto &r) {
    auto &block_sizes = block_sizes_ets.local();
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      ++block_sizes[p_graph.block(u)];
    }
  });
  const auto block_sizes = block_sizes_ets.combine(std::plus{});

  std::vector<NodeID> block_offsets(p_graph.k() + 1);
  parallel::prefix_sum(block_sizes.begin(), block_sizes.end(), block_offsets.begin() + 1);

  // Map blocks to the PE from which we use the partition
  NoinitVector<PEID> block_owner(p_graph.k());
  tbb::parallel_for<BlockID>(0, p_graph.k(), [&](const BlockID b) {
    block_owner[b] = find_winner(b);
  });

  // Assign nodes in p_graph to new blocks
  auto partition = p_graph.take_partition(); // NOTE: do not use p_graph after this

  p_graph.pfor_nodes([&](const NodeID u) {
    const BlockID block = partition[u];
    const PEID owner = block_owner[block];
    const BlockID first_block_on_owner = offsets.first_block_on_pe(owner);
    const BlockID block_offset = block_offsets[block] - block_offsets[first_block_on_owner];
    const NodeID mapped_u = mapping[u]; // ID of u in its block-induced subgraph
    const BlockID new_b = block * k_multiplier + partition_recvbufs[owner][mapped_u + block_offset];
    partition[u] = new_b;
  });

  // Create partitioned graph with the new partition
  DistributedPartitionedGraph new_p_graph(&p_graph.graph(), new_k, std::move(partition));

  // Synchronize block assignment of ghost nodes
  synchronize_ghost_node_block_ids(new_p_graph);

  KASSERT(
      debug::validate_partition(new_p_graph),
      "graph partition in inconsistent state",
      assert::heavy
  );
  return new_p_graph;
}
} // namespace kaminpar::dist::graph
