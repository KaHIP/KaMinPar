/*******************************************************************************
 * Wrapper class that delegates all function calls to a concrete graph object.
 *
 * Most function calls are resolved via dynamic binding. Thus, they should not
 * be used when performance is critical. Instead, use an downcast and templatize
 * tight loops.
 *
 * @file:   distributed_graph.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#include "kaminpar-dist/datastructures/distributed_graph.h"

#include <iomanip>
#include <numeric>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/graphutils/communication.h"
#include "kaminpar-dist/logger.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/vector_ets.h"

namespace {
template <typename R> bool all_equal(const R &r) {
  return std::adjacent_find(r.begin(), r.end(), std::not_equal_to{}) == r.end();
}
} // namespace

namespace kaminpar::dist {

void print_graph_summary(const DistributedGraph &graph) {
  const auto [n_min, n_avg, n_max, n_sum] = mpi::gather_statistics(graph.n(), graph.communicator());
  const double n_imbalance = 1.0 * n_max / n_avg;
  const auto [ghost_n_min, ghost_n_avg, ghost_n_max, ghost_n_sum] =
      mpi::gather_statistics(graph.ghost_n(), graph.communicator());
  const double ghost_n_imbalance = 1.0 * ghost_n_max / ghost_n_avg;
  const auto [m_min, m_avg, m_max, m_sum] = mpi::gather_statistics(graph.m(), graph.communicator());
  const double m_imbalance = 1.0 * m_max / m_avg;
  const auto [max_node_weight_min, max_node_weight_avg, max_node_weight_max, max_node_weight_sum] =
      mpi::gather_statistics(graph.max_node_weight(), graph.communicator());

  LOG << "  Number of nodes: " << graph.global_n() << " | Number of edges: " << graph.global_m();
  LOG << "  Number of local nodes: [Min=" << n_min << " | Mean=" << static_cast<NodeID>(n_avg)
      << " | Max=" << n_max << " | Imbalance=" << n_imbalance << "]";
  LOG << "  Number of ghost nodes: [Min=" << ghost_n_min
      << " | Mean=" << static_cast<NodeID>(ghost_n_avg) << " | Max=" << ghost_n_max
      << " | Imbalance=" << ghost_n_imbalance << "]";
  LOG << "  Number of edges:       [Min=" << m_min << " | Mean=" << static_cast<EdgeID>(m_avg)
      << " | Max=" << m_max << " | Imbalance=" << m_imbalance << "]";
  LOG << "  Maximum node weight:   [Min=" << max_node_weight_min
      << " | Mean=" << static_cast<NodeWeight>(max_node_weight_avg)
      << " | Max=" << max_node_weight_max << "]";
}

namespace debug {
void print_graph(const DistributedGraph &graph) {
  std::ostringstream buf;

  const int w = std::ceil(std::log10(graph.global_n()));

  buf << "Distributed graph on " << mpi::get_comm_rank(graph.communicator())
      << " PEs, n=" << graph.n() << " m=" << graph.m() << " ghost_n=" << graph.ghost_n()
      << " total_n=" << graph.total_n() << " global_n=" << graph.global_n()
      << " global_m=" << graph.global_m() << "\n";
  buf << "--------------------------------------------------------------------------------\n";
  for (const NodeID u : graph.all_nodes()) {
    const char u_prefix = graph.is_owned_node(u) ? ' ' : '!';
    buf << u_prefix << "L" << std::setw(w) << u << " G" << std::setw(w)
        << graph.local_to_global_node(u) << " NW" << std::setw(w) << graph.node_weight(u);

    if (graph.is_owned_node(u)) {
      buf << " | ";
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        const char v_prefix = graph.is_owned_node(v) ? ' ' : '!';
        buf << v_prefix << "L" << std::setw(w) << v << " G" << std::setw(w)
            << graph.local_to_global_node(v) << " EW" << std::setw(w) << w << " NW" << std::setw(w)
            << graph.node_weight(v) << "\t";
      });
      if (graph.degree(u) == 0) {
        buf << "<isolated>";
      }
    }
    buf << "\n";
  }
  buf << "--------------------------------------------------------------------------------\n";
  buf << "  where L = local ID, G = global ID, NW = node weight, EW = edge weight\n";
  SLOG << buf.str();
}

void print_local_graph_stats(const DistributedGraph &graph) {
  std::stringstream ss;
  ss << "n=" << graph.n() << " ";
  ss << "total_n=" << graph.total_n() << " ";
  ss << "ghost_n=" << graph.ghost_n() << " ";
  ss << "m=" << graph.m() << " ";

  std::array<EdgeID, 32> buckets;
  std::fill(buckets.begin(), buckets.end(), 0);

  EdgeID local_m = 0, nonlocal_m = 0;
  NodeID min_deg = std::numeric_limits<NodeID>::max(), max_deg = 0;
  for (NodeID u = 0; u < graph.n(); ++u) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
      if (graph.is_owned_node(v)) {
        ++local_m;
      } else {
        ++nonlocal_m;
      }
    });
    if (graph.degree(u) == 0) {
      ++buckets[0];
    } else {
      ++buckets[std::min<int>(buckets.size() - 1, 1 + std::log2(graph.degree(u)))];
    }
    min_deg = std::min(graph.degree(u), min_deg);
    max_deg = std::max(graph.degree(u), max_deg);
  }

  ss << "local_m=" << local_m << " ";
  ss << "nonlocal_m=" << nonlocal_m << " ";
  ss << "min_deg=" << min_deg << " ";
  ss << "max_deg = " << max_deg << " ";
  for (std::size_t i = 0; i < buckets.size(); ++i) {
    if (buckets[i] > 0) {
      ss << "deg_" << i << "=" << buckets[i] << " ";
    }
  }

  DLOG << ss.str();
}

bool validate_graph(const DistributedGraph &graph) {
  MPI_Comm comm = graph.communicator();

  const PEID size = mpi::get_comm_size(comm);
  const PEID rank = mpi::get_comm_rank(comm);

  {
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
    for (const NodeID ghost : graph.ghost_nodes()) {
      if (graph.ghost_owner(ghost) == rank) {
        LOG_ERROR << "on PE " << rank << ": local owner of ghost node " << ghost;
        return false;
      }
    }
  }

  mpi::barrier(comm);

  {
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
    struct GhostNodeEdge {
      GlobalNodeID owned;
      GlobalNodeID ghost;
    };

    const auto recvbufs = mpi::graph::sparse_alltoall_interface_to_ghost_get<GhostNodeEdge>(
        graph,
        [&](const NodeID u, EdgeID, const NodeID v, EdgeWeight) -> GhostNodeEdge {
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
        graph.adjacent_nodes(local_owned_node, [&](const NodeID v) {
          if (v == local_ghost_node) {
            found = true;
            return true;
          }

          return false;
        });
        if (!found) {
          LOG_ERROR << "PE " << pe << " expects a local edge " << local_owned_node
                    << " (owned, global node " << owned << ") --> " << local_ghost_node
                    << " (ghost, global node " << ghost
                    << ") on this PE, but the edge does not exist";
          LOG_ERROR << "Outgoing edges from local node " << local_owned_node << " are:";

          graph.adjacent_nodes(local_owned_node, [&](const NodeID v) {
            LOG_ERROR << "\t- " << v << " (global " << graph.local_to_global_node(v) << ")";
          });
          return false;
        }
      }
    }
  }

  mpi::barrier(comm);

  if (graph.sorted()) {
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
} // namespace debug
} // namespace kaminpar::dist
