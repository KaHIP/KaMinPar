/*******************************************************************************
 * Static distributed compressed graph data structure.
 *
 * @file:   distributed_compressed_graph.cc
 * @author: Daniel Salwasser
 * @date:   07.06.2024
 ******************************************************************************/
#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"

#include "kaminpar-dist/graphutils/communication.h"

#include "kaminpar-common/parallel/vector_ets.h"

namespace kaminpar::dist {

void DistributedCompressedGraph::init_high_degree_info(const EdgeID high_degree_threshold) const {
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
          const NodeID local_node = map_remote_node(remote_node, pe);
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

void DistributedCompressedGraph::init_degree_buckets() {
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

void DistributedCompressedGraph::init_total_weights() {
  if (is_node_weighted()) {
    const auto begin_node_weights = _node_weights.begin();
    const auto end_node_weights = begin_node_weights + static_cast<std::size_t>(n());

    _total_node_weight = parallel::accumulate(begin_node_weights, end_node_weights, 0);
    _max_node_weight = parallel::max_element(begin_node_weights, end_node_weights);
  } else {
    _total_node_weight = n();
    _max_node_weight = 1;
  }

  _global_total_node_weight =
      mpi::allreduce<GlobalNodeWeight>(_total_node_weight, MPI_SUM, communicator());
  _global_max_node_weight =
      mpi::allreduce<GlobalNodeWeight>(_max_node_weight, MPI_MAX, communicator());
  _global_total_edge_weight = mpi::allreduce<GlobalEdgeWeight>(
      _compressed_neighborhoods.total_edge_weight(), MPI_SUM, communicator()
  );
}

void DistributedCompressedGraph::init_communication_metrics() {
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
      adjacent_nodes(u, [&](const NodeID v) {
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
      });
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

} // namespace kaminpar::dist
