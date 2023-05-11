/*******************************************************************************
 * @file:   distributed_graph.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Static distributed graph data structure.
 ******************************************************************************/
#include "dkaminpar/datastructures/distributed_graph.h"

#include <iomanip>
#include <numeric>

#include "dkaminpar/graphutils/communication.h"
#include "dkaminpar/mpi/wrapper.h"

#include "common/datastructures/marker.h"
#include "common/math.h"
#include "common/parallel/algorithm.h"
#include "common/parallel/vector_ets.h"
#include "common/scalable_vector.h"
#include "common/timer.h"

namespace kaminpar::dist {
void DistributedGraph::print() const {
  std::ostringstream buf;

  const int w = std::ceil(std::log10(global_n()));

  buf << "n=" << n() << " m=" << m() << " ghost_n=" << ghost_n() << " total_n=" << total_n()
      << "\n";
  buf << "---------------------------------------------------------------------"
         "-----------\n";
  for (const NodeID u : all_nodes()) {
    const char u_prefix = is_owned_node(u) ? ' ' : '!';
    buf << u_prefix << "L" << std::setw(w) << u << " G" << std::setw(w) << local_to_global_node(u)
        << " NW" << std::setw(w) << node_weight(u);

    if (is_owned_node(u)) {
      buf << " | ";
      for (const auto [e, v] : neighbors(u)) {
        const char v_prefix = is_owned_node(v) ? ' ' : '!';
        buf << v_prefix << "L" << std::setw(w) << v << " G" << std::setw(w)
            << local_to_global_node(v) << " EW" << std::setw(w) << edge_weight(e) << " NW"
            << std::setw(w) << node_weight(v) << "\t";
      }
      if (degree(u) == 0) {
        buf << "<empty>";
      }
    }
    buf << "\n";
  }
  buf << "---------------------------------------------------------------------"
         "-----------\n";
  SLOG << buf.str();
}

void DistributedGraph::init_high_degree_info(const EdgeID high_degree_threshold) const {
  if (_high_degree_threshold == high_degree_threshold) {
    return;
  }

  _high_degree_threshold = high_degree_threshold;
  _high_degree_ghost_node.resize(ghost_n());

  struct Message {
    NodeID node;
    std::uint8_t high_degree;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message>(
      *this,
      [&](const NodeID u) -> Message {
        return {.node = u, .high_degree = degree(u) > _high_degree_threshold};
      },
      [&](const auto &recv_buffer, const PEID pe) {
        tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
          const auto &[remote_node, high_degree] = recv_buffer[i];
          const NodeID local_node = remote_to_local_node(remote_node, pe);
          _high_degree_ghost_node[local_node - n()] = high_degree;
        });
      }
  );
}

namespace {
inline EdgeID degree_bucket(const EdgeID degree) {
  return (degree == 0) ? 0 : math::floor_log2(degree) + 1;
}
} // namespace

void DistributedGraph::init_degree_buckets() {
  KASSERT(std::all_of(_buckets.begin(), _buckets.end(), [](const auto n) { return n == 0; }));

  if (_sorted) {
    parallel::vector_ets<NodeID> buckets_ets(_buckets.size());
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, n()), [&](const auto &r) {
      auto &buckets = buckets_ets.local();
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        auto bucket = degree_bucket(degree(u)) + 1;
        ++buckets[bucket];
      }
    });
    const auto buckets = buckets_ets.combine(std::plus{});
    std::copy(buckets.begin(), buckets.end(), _buckets.begin());

    auto last_nonempty_bucket =
        std::find_if(_buckets.rbegin(), _buckets.rend(), [](const auto n) { return n > 0; });
    _number_of_buckets = std::distance(_buckets.begin(), (last_nonempty_bucket + 1).base());
  } else {
    _buckets[1] = n();
    _number_of_buckets = 1;
  }

  std::partial_sum(_buckets.begin(), _buckets.end(), _buckets.begin());
}

void DistributedGraph::init_total_weights() {
  if (is_node_weighted()) {
    const auto begin_node_weights = _node_weights.begin();
    const auto end_node_weights = begin_node_weights + static_cast<std::size_t>(n());

    _total_node_weight = parallel::accumulate(begin_node_weights, end_node_weights, 0);
    _max_node_weight = parallel::max_element(begin_node_weights, end_node_weights);
  } else {
    _total_node_weight = n();
    _max_node_weight = 1;
  }

  if (is_edge_weighted()) {
    _total_edge_weight = parallel::accumulate(_edge_weights.begin(), _edge_weights.end(), 0);
  } else {
    _total_edge_weight = m();
  }

  _global_total_node_weight =
      mpi::allreduce<GlobalNodeWeight>(_total_node_weight, MPI_SUM, communicator());
  _global_max_node_weight =
      mpi::allreduce<GlobalNodeWeight>(_max_node_weight, MPI_MAX, communicator());
  _global_total_edge_weight =
      mpi::allreduce<GlobalEdgeWeight>(_total_edge_weight, MPI_SUM, communicator());
}

void DistributedGraph::init_communication_metrics() {
  const PEID size = mpi::get_comm_size(_communicator);

  tbb::enumerable_thread_specific<std::vector<EdgeID>> edge_cut_to_pe_ets{[&] {
    return std::vector<EdgeID>(size);
  }};
  tbb::enumerable_thread_specific<std::vector<EdgeID>> comm_vol_to_pe_ets{[&] {
    return std::vector<EdgeID>(size);
  }};

  pfor_nodes_range([&](const auto r) {
    auto &edge_cut_to_pe = edge_cut_to_pe_ets.local();
    auto &comm_vol_to_pe = comm_vol_to_pe_ets.local();
    Marker<> counted_pe{static_cast<std::size_t>(size)};

    for (NodeID u = r.begin(); u < r.end(); ++u) {
      for (const auto v : adjacent_nodes(u)) {
        if (is_ghost_node(v)) {
          const PEID owner = ghost_owner(v);
          KASSERT(static_cast<std::size_t>(owner) < edge_cut_to_pe.size());
          ++edge_cut_to_pe[owner];

          if (!counted_pe.get(owner)) {
            KASSERT(static_cast<std::size_t>(owner) < counted_pe.size());
            counted_pe.set(owner);

            KASSERT(static_cast<std::size_t>(owner) < comm_vol_to_pe.size());
            ++comm_vol_to_pe[owner];
          }
        }
      }
      counted_pe.reset();
    }
  });

  _edge_cut_to_pe.clear();
  _edge_cut_to_pe.resize(size);
  for (const auto &edge_cut_to_pe : edge_cut_to_pe_ets) { // PE x THREADS
    for (std::size_t i = 0; i < edge_cut_to_pe.size(); ++i) {
      _edge_cut_to_pe[i] += edge_cut_to_pe[i];
    }
  }

  _comm_vol_to_pe.clear();
  _comm_vol_to_pe.resize(size);
  for (const auto &comm_vol_to_pe : comm_vol_to_pe_ets) {
    for (std::size_t i = 0; i < comm_vol_to_pe.size(); ++i) {
      _comm_vol_to_pe[i] += comm_vol_to_pe[i];
    }
  }
}

void DistributedPartitionedGraph::init_block_weights() {
  parallel::vector_ets<BlockWeight> local_block_weights_ets(k());

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n()), [&](const auto &r) {
    auto &local_block_weights = local_block_weights_ets.local();
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      local_block_weights[block(u)] += node_weight(u);
    }
  });
  auto local_block_weights = local_block_weights_ets.combine(std::plus{});

  scalable_vector<BlockWeight> global_block_weights_nonatomic(k());
  mpi::allreduce(
      local_block_weights.data(),
      global_block_weights_nonatomic.data(),
      k(),
      MPI_SUM,
      communicator()
  );

  _block_weights.resize(k());
  pfor_blocks([&](const BlockID b) { _block_weights[b] = global_block_weights_nonatomic[b]; });
}

namespace graph {
void print_summary(const DistributedGraph &graph) {
  const auto global_n = graph.global_n();
  const auto global_m = graph.global_m();
  const auto [local_n_min, local_n_avg, local_n_max, local_n_sum] =
      mpi::gather_statistics(graph.n(), graph.communicator());
  const double local_n_imbalance = 1.0 * local_n_max / local_n_avg;
  const auto [local_m_min, local_m_avg, local_m_max, local_m_sum] =
      mpi::gather_statistics(graph.m(), graph.communicator());
  const double local_m_imbalance = 1.0 * local_m_max / local_m_avg;
  const auto [ghost_min, ghost_avg, ghost_max, ghost_sum] =
      mpi::gather_statistics(graph.ghost_n(), graph.communicator());
  const double ghost_imbalance = 1.0 * ghost_max / ghost_avg;

  const auto local_width =
      static_cast<std::streamsize>(std::log10(std::max({local_n_max, local_m_max, ghost_max})) + 1);

  LOG << "  Global number of nodes: " << global_n;
  LOG << "  Global number of edges: " << global_m;
  LOG << "  Local number of nodes:  [min=" << std::setw(local_width) << local_n_min
      << "|avg=" << std::fixed << std::setprecision(1) << std::setw(local_width) << local_n_avg
      << "|max=" << std::setw(local_width) << local_n_max << "|imbalance=" << std::fixed
      << std::setprecision(3) << std::setw(local_width) << local_n_imbalance << "]";
  LOG << "  Local number of edges:  [min=" << std::setw(local_width) << local_m_min
      << "|avg=" << std::fixed << std::setprecision(1) << std::setw(local_width) << local_m_avg
      << "|max=" << std::setw(local_width) << local_m_max << "|imbalance=" << std::fixed
      << std::setprecision(3) << std::setw(local_width) << local_m_imbalance << "]";
  LOG << "  Number of ghost nodes:  [min=" << std::setw(local_width) << ghost_min
      << "|avg=" << std::fixed << std::setprecision(1) << std::setw(local_width) << ghost_avg
      << "|max=" << std::setw(local_width) << ghost_max << "|imbalance=" << std::fixed
      << std::setprecision(3) << std::setw(local_width) << ghost_imbalance << "]";
}
} // namespace graph

namespace graph::debug {
SET_DEBUG(false);

namespace {
template <typename R> bool all_equal(const R &r) {
  return std::adjacent_find(r.begin(), r.end(), std::not_equal_to{}) == r.end();
}
} // namespace

bool validate(const DistributedGraph &graph) {
  MPI_Comm comm = graph.communicator();

  const PEID size = mpi::get_comm_size(comm);
  const PEID rank = mpi::get_comm_rank(comm);

  {
    DBG << "Checking global number of nodes and edges ...";

    const GlobalNodeID root_global_n = mpi::bcast(graph.global_n(), 0, comm);
    const GlobalEdgeID root_global_m = mpi::bcast(graph.global_m(), 0, comm);
    if (root_global_n != graph.global_n()) {
      LOG_ERROR << "on PE " << rank << ": inconsistent global number of nodes (" << graph.global_n()
                << ") with root (" << root_global_n << ")";
      return false;
    }
    if (root_global_m != graph.global_m()) {
      LOG_ERROR << "on PE " << rank << ": inconsistent global number of edges (" << graph.global_m()
                << ") with root (" << root_global_m << ")";
      return false;
    }
  }

  mpi::barrier(comm);

  {
    DBG << "Checking node distribution ...";

    if (graph.node_distribution().size() != static_cast<std::size_t>(size + 1)) {
      LOG_ERROR << "on PE " << rank << ": expected size of the node distribution array to be "
                << size + 1 << ", but is " << graph.node_distribution().size();
      return false;
    }
    if (graph.edge_distribution().size() != static_cast<std::size_t>(size + 1)) {
      LOG_ERROR << "on PE " << rank << ": expected size of the edge distribution array to be "
                << size + 1 << ", but is " << graph.edge_distribution().size();
      return false;
    }
    if (graph.node_distribution().front() != 0u) {
      LOG_ERROR << "on PE " << rank << ": node distribution starts at "
                << graph.node_distribution().front();
      return false;
    }
    if (graph.edge_distribution().front() != 0u) {
      LOG_ERROR << "on PE " << rank << ": edge distribution starts at "
                << graph.edge_distribution().front();
      return false;
    }
    if (graph.edge_distribution().back() != graph.global_m()) {
      LOG_ERROR << "on PE " << rank << ": edge distribution ends at "
                << graph.edge_distribution().back() << ", but graph has " << graph.global_m()
                << " edges";
      return false;
    }
    if (graph.n() != graph.node_distribution(rank + 1) - graph.node_distribution(rank)) {
      LOG_ERROR << "on PE " << rank << ": local graph has " << graph.n()
                << " nodes, but node distribution array claims "
                << graph.node_distribution(rank + 1) - graph.node_distribution(rank) << " nodes";
      return false;
    }
    if (graph.m() != graph.edge_distribution(rank + 1) - graph.edge_distribution(rank)) {
      LOG_ERROR << "on PE " << rank << ": local graph has " << graph.m()
                << " nodes, but edge distribution array claims "
                << graph.edge_distribution(rank + 1) - graph.edge_distribution(rank) << " nodes";
      return false;
    }

    for (PEID pe = 1; pe < size + 1; ++pe) {
      const GlobalNodeID nodes_entry_on_root = mpi::bcast(graph.node_distribution(pe), 0, comm);
      const GlobalEdgeID edges_entry_on_root = mpi::bcast(graph.edge_distribution(pe), 0, comm);

      if (graph.node_distribution(pe) != nodes_entry_on_root) {
        LOG_ERROR << "on PE " << rank << ": inconsistent entry " << graph.node_distribution(pe)
                  << " in the node distribution array, expected " << nodes_entry_on_root;
        return false;
      }
      if (graph.edge_distribution(pe) != edges_entry_on_root) {
        LOG_ERROR << "on PE " << rank << ": inconsistent entry " << graph.edge_distribution(pe)
                  << " in the edge distribution array, expected " << edges_entry_on_root;
        return false;
      }
    }
  }

  mpi::barrier(comm);

  {
    DBG << "Checking that owners of ghost nodes are not local ...";
    for (const NodeID ghost : graph.ghost_nodes()) {
      if (graph.ghost_owner(ghost) == rank) {
        LOG_ERROR << "on PE " << rank << ": local owner of ghost node " << ghost;
        return false;
      }
    }
  }

  mpi::barrier(comm);

  {
    DBG << "Checking node weights of ghost nodes ...";

    struct GhostNodeWeightMessage {
      GlobalNodeID node;
      NodeWeight weight;
    };

    const auto recvbufs = mpi::graph::sparse_alltoall_interface_to_pe_get<GhostNodeWeightMessage>(
        graph,
        [&](const NodeID u) -> GhostNodeWeightMessage {
          return {.node = graph.local_to_global_node(u), .weight = graph.node_weight(u)};
        }
    );

    for (PEID pe = 0; pe < size; ++pe) {
      for (const auto &[node, weight] : recvbufs[pe]) {
        if (!graph.contains_global_node(node)) {
          LOG_ERROR << "on PE " << rank << ": invalid global node " << node << " with weight "
                    << weight << " sent from PE " << pe;
          return false;
        }
      }
    }
  }

  mpi::barrier(comm);

  {
    DBG << "Checking edges to ghost nodes ...";

    struct GhostNodeEdge {
      GlobalNodeID owned;
      GlobalNodeID ghost;
    };

    const auto recvbufs = mpi::graph::sparse_alltoall_interface_to_ghost_get<GhostNodeEdge>(
        graph,
        [&](const NodeID u, const EdgeID, const NodeID v) -> GhostNodeEdge {
          return {.owned = graph.local_to_global_node(u), .ghost = graph.local_to_global_node(v)};
        }
    );

    for (PEID pe = 0; pe < size; ++pe) {
      for (const auto &[ghost, owned] : recvbufs[pe]) {
        if (!graph.contains_global_node(ghost)) {
          LOG_ERROR << "global node " << ghost
                    << " does not exist as ghost node on this PE (expected by PE " << pe << ")";
          return false;
        }
        if (!graph.contains_global_node(owned)) {
          LOG_ERROR << "global node " << owned
                    << " does not exist on this PE (expected to be owned by "
                       "this PE by PE "
                    << pe << ")";
          return false;
        }

        const NodeID local_owned_node = graph.global_to_local_node(owned);
        const NodeID local_ghost_node = graph.global_to_local_node(ghost);

        if (local_owned_node >= graph.n()) {
          LOG_ERROR << "global node " << owned << " (local " << local_owned_node
                    << ") is expected to be owned by this PE (by PE " << pe << "), but it is not";
          return false;
        }
        if (local_ghost_node < graph.n()) {
          LOG_ERROR << "global node " << ghost << " (local " << local_ghost_node
                    << ") is expected to be a ghost node on this PE, but it is "
                       "a owned node (expected by PE "
                    << pe << "); number of local nodes: " << graph.n();
          return false;
        }

        bool found = false;
        for (const auto v : graph.adjacent_nodes(local_owned_node)) {
          if (v == local_ghost_node) {
            found = true;
            break;
          }
        }
        if (!found) {
          LOG_ERROR << "PE " << pe << " expects a local edge " << local_owned_node
                    << " (owned, global node " << owned << ") --> " << local_ghost_node
                    << " (ghost, global node " << ghost
                    << ") on this PE, but the edge does not exist";
          LOG_ERROR << "Outgoing edges from local node " << local_owned_node << " are:";
          for (const auto v : graph.adjacent_nodes(local_owned_node)) {
            LOG_ERROR << "\t- " << v << " (global " << graph.local_to_global_node(v) << ")";
          }
          return false;
        }
      }
    }
  }

  mpi::barrier(comm);

  if (graph.sorted()) {
    DBG << "Checking degree buckets ...";

    for (std::size_t bucket = 0; bucket < graph.number_of_buckets(); ++bucket) {
      if (graph.bucket_size(bucket) == 0) {
        continue;
      }

      if (graph.first_node_in_bucket(bucket) == graph.first_invalid_node_in_bucket(bucket)) {
        LOG_ERROR << "on PE " << rank
                  << ": bucket is empty, but graph data structure claims that it "
                     "has size "
                  << graph.bucket_size(bucket);
        return false;
      }

      for (NodeID u = graph.first_node_in_bucket(bucket);
           u < graph.first_invalid_node_in_bucket(bucket);
           ++u) {
        const auto expected_bucket = degree_bucket(graph.degree(u));
        if (expected_bucket != bucket) {
          LOG_ERROR << "on PE " << rank << ":node " << u << " with degree " << graph.degree(u)
                    << " is expected to be in bucket " << expected_bucket << ", but is in bucket "
                    << bucket;
          return false;
        }
      }
    }
  }

  mpi::barrier(comm);
  return true;
}

bool validate_partition(const DistributedPartitionedGraph &p_graph) {
  MPI_Comm comm = p_graph.communicator();

  const PEID size = mpi::get_comm_size(comm);
  const PEID rank = mpi::get_comm_rank(comm);

  {
    DBG << "Checking block counts ...";

    const BlockID root_k = mpi::bcast(p_graph.k(), 0, comm);
    if (root_k != p_graph.k()) {
      LOG_ERROR << "on PE " << rank << ": number of blocks (" << p_graph.k()
                << ") differs from number of blocks on the root PE (" << root_k << ")";
      return false;
    }
  }

  mpi::barrier(comm);

  {
    DBG << "Checking block IDs ...";

    for (const NodeID u : p_graph.all_nodes()) {
      if (p_graph.block(u) >= p_graph.k()) {
        LOG_ERROR << "on PE " << rank << ": node " << u << " assigned to invalid block "
                  << p_graph.block(u);
        return false;
      }
    }
  }

  mpi::barrier(comm);

  {
    DBG << "Checking block weights ...";

    StaticArray<BlockWeight> recomputed_block_weights(p_graph.k());
    for (const NodeID u : p_graph.nodes()) {
      recomputed_block_weights[p_graph.block(u)] += p_graph.node_weight(u);
    }
    MPI_Allreduce(
        MPI_IN_PLACE,
        recomputed_block_weights.data(),
        asserting_cast<int>(p_graph.k()),
        mpi::type::get<BlockWeight>(),
        MPI_SUM,
        comm
    );

    for (const BlockID b : p_graph.blocks()) {
      if (p_graph.block_weight(b) != recomputed_block_weights[b]) {
        LOG_ERROR << "on PE " << rank << ": expected weight of block " << b << " is "
                  << p_graph.block_weight(b) << ", but actual weight is "
                  << recomputed_block_weights[b];
        return false;
      }
    }

    mpi::barrier(comm);
  }

  {
    DBG << "Checking ghost node block assignments ...";

    // Build a global partition array on the root PE
    StaticArray<BlockID> global_partition(0);
    if (rank == 0) {
      global_partition.resize(p_graph.global_n());
    }

    std::vector<int> displs(p_graph.node_distribution().begin(), p_graph.node_distribution().end());
    std::vector<int> counts(size);
    for (PEID pe = 0; pe < size; ++pe) {
      counts[pe] = displs[pe + 1] - displs[pe];
    }

    MPI_Gatherv(
        p_graph.partition().data(),
        asserting_cast<int>(p_graph.n()),
        mpi::type::get<BlockID>(),
        global_partition.data(),
        counts.data(),
        displs.data(),
        mpi::type::get<BlockID>(),
        0,
        comm
    );

    // Collect ghost node assignments on the root PE
    struct NodeAssignment {
      GlobalNodeID node;
      BlockID block;
    };
    std::vector<NodeAssignment> sendbuf;
    for (const NodeID ghost_u : p_graph.ghost_nodes()) {
      const GlobalNodeID global = p_graph.local_to_global_node(ghost_u);
      const BlockID block = p_graph.block(ghost_u);

      if (rank == 0) {
        if (global_partition[global] != block) {
          KASSERT(global < global_partition.size());
          LOG_ERROR << "inconsistent assignment of node " << global
                    << " on PE 0: local assignment to block " << block
                    << " inconsistent with actual assignment to block " << global_partition[global];
          return false;
        }
      } else {
        sendbuf.push_back({global, block});
      }
    }

    if (rank == 0) {
      for (int pe = 1; pe < size; ++pe) {
        const auto recvbuf = mpi::probe_recv<NodeAssignment>(pe, 0, comm);

        for (const auto &[global, block] : recvbuf) {
          if (global_partition[global] != block) {
            LOG_ERROR << "inconsistent assignment of node " << global
                      << " on PE 0: local assignment to block " << block
                      << " inconsistent with actual assignment to block "
                      << global_partition[global];
            return false;
          }
        }
      }
    } else {
      mpi::send(sendbuf.data(), sendbuf.size(), 0, 0, comm);
    }
  }

  mpi::barrier(comm);

  return true;
}
} // namespace graph::debug
} // namespace kaminpar::dist
