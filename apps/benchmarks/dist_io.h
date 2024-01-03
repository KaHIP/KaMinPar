/*******************************************************************************
 * @file:   dist_io.h
 * @author: Daniel Seemaier
 * @date:   13.06.2023
 * @brief:  Common KaGen-based IO code for benchmarks.
 ******************************************************************************/
#pragma once

#include <kaminpar-dist/datastructures/distributed_graph.h>
#include <kaminpar-dist/datastructures/ghost_node_mapper.h>
#include <kaminpar-dist/graphutils/synchronization.h>
#include <kaminpar-dist/metrics.h>
#include <mpi.h>
#include <tbb/parallel_invoke.h>

#include "apps/benchmarks/shm_io.h"

namespace kaminpar::dist {
struct DistributedGraphWrapper {
  std::unique_ptr<DistributedGraph> graph;
};

struct DistributedPartitionedGraphWrapper : public DistributedGraphWrapper {
  std::unique_ptr<DistributedPartitionedGraph> p_graph;
};

inline DistributedGraphWrapper load_graph(const std::string &graph_name) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

  kagen::Graph kagen_graph = invoke_kagen(graph_name);
  auto xadj = kagen_graph.TakeXadj<>();
  auto adjncy = kagen_graph.TakeAdjncy<>();
  auto vwgt = kagen_graph.TakeVertexWeights<>();
  auto adjwgt = kagen_graph.TakeEdgeWeights<>();

  auto vtxdist =
      kagen::BuildVertexDistribution<unsigned long>(kagen_graph, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

  const NodeID n = static_cast<NodeID>(vtxdist[rank + 1] - vtxdist[rank]);
  const GlobalNodeID from = vtxdist[rank];
  const GlobalNodeID to = vtxdist[rank + 1];
  const EdgeID m = static_cast<EdgeID>(xadj[n]);

  StaticArray<GlobalNodeID> node_distribution(vtxdist.begin(), vtxdist.end());
  StaticArray<GlobalEdgeID> edge_distribution(size + 1);
  edge_distribution[rank] = m;
  MPI_Allgather(
      MPI_IN_PLACE,
      1,
      mpi::type::get<GlobalEdgeID>(),
      edge_distribution.data(),
      1,
      mpi::type::get<GlobalEdgeID>(),
      MPI_COMM_WORLD
  );
  std::exclusive_scan(
      edge_distribution.begin(),
      edge_distribution.end(),
      edge_distribution.begin(),
      static_cast<GlobalEdgeID>(0)
  );

  StaticArray<EdgeID> nodes;
  StaticArray<NodeID> edges;
  StaticArray<NodeWeight> node_weights;
  StaticArray<EdgeWeight> edge_weights;
  graph::GhostNodeMapper mapper(rank, node_distribution);

  tbb::parallel_invoke(
      [&] {
        nodes.resize(n + 1);
        tbb::parallel_for<NodeID>(0, n + 1, [&](const NodeID u) { nodes[u] = xadj[u]; });
      },
      [&] {
        edges.resize(m);
        tbb::parallel_for<EdgeID>(0, m, [&](const EdgeID e) {
          const GlobalNodeID v = adjncy[e];
          if (v >= from && v < to) {
            edges[e] = static_cast<NodeID>(v - from);
          } else {
            edges[e] = mapper.new_ghost_node(v);
          }
        });
      },
      [&] {
        if (vwgt.empty()) {
          return;
        }
        node_weights.resize(n);
        tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) { node_weights[u] = vwgt[u]; });
      },
      [&] {
        if (adjwgt.empty()) {
          return;
        }
        edge_weights.resize(m);
        tbb::parallel_for<EdgeID>(0, m, [&](const EdgeID e) { edge_weights[e] = adjwgt[e]; });
      }
  );

  auto [global_to_ghost, ghost_to_global, ghost_owner] = mapper.finalize();

  DistributedGraphWrapper wrapper;
  wrapper.graph = std::make_unique<DistributedGraph>(
      std::move(node_distribution),
      std::move(edge_distribution),
      std::move(nodes),
      std::move(edges),
      std::move(node_weights),
      std::move(edge_weights),
      std::move(ghost_owner),
      std::move(ghost_to_global),
      std::move(global_to_ghost),
      false,
      MPI_COMM_WORLD
  );
  return wrapper;
}

template <typename Container>
Container load_vector(
    const std::string &filename,
    const GlobalNodeID from,
    const GlobalNodeID to,
    const GlobalNodeID size
) {
  Container vec(size);

  std::ifstream in(filename);
  if (!in) {
    throw std::runtime_error("could not open file " + filename);
  }

  for (GlobalNodeID u = 0; u < to; ++u) {
    std::uint64_t value;
    if (!(in >> value)) {
      throw std::runtime_error("file " + filename + " does not contain enough values");
    }
    if (u >= from) {
      vec[u - from] = static_cast<typename Container::value_type>(value);
    }
  }

  return vec;
}

template <typename Container>
Container load_node_property_vector(const DistributedGraph &graph, const std::string &filename) {
  Container vec = load_vector<Container>(
      filename, graph.offset_n(), graph.offset_n() + graph.n(), graph.total_n()
  );

  struct Message {
    NodeID node;
    typename Container::value_type value;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message>(
      graph,
      [&](const NodeID u) -> Message { return {.node = u, .value = vec[u]}; },
      [&](const auto &recv_buffer, const PEID pe) {
        tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
          const auto [their_lnode, value] = recv_buffer[i];
          const auto gnode = static_cast<GlobalNodeID>(graph.offset_n(pe) + their_lnode);
          const NodeID lnode = graph.global_to_local_node(gnode);
          vec[lnode] = value;
        });
      }
  );

  return vec;
}

inline DistributedPartitionedGraphWrapper
load_partitioned_graph(const std::string &graph_name, const std::string &partition_name) {
  DistributedGraphWrapper graph_wrapper = load_graph(graph_name);

  DistributedPartitionedGraphWrapper wrapper;
  wrapper.graph = std::move(graph_wrapper.graph);
  auto &graph = wrapper.graph;

  const GlobalNodeID first_node = graph->offset_n();
  const GlobalNodeID first_invalid_node = graph->offset_n() + graph->n();

  auto partition = load_node_property_vector<StaticArray<BlockID>>(*graph, partition_name);

  BlockID k = *std::max_element(partition.begin(), partition.end()) + 1;
  MPI_Allreduce(MPI_IN_PLACE, &k, 1, mpi::type::get<BlockID>(), MPI_MAX, MPI_COMM_WORLD);

  StaticArray<BlockWeight> block_weights(k);
  for (const NodeID u : graph->nodes()) {
    block_weights[partition[u]] += graph->node_weight(u);
  }
  MPI_Allreduce(
      MPI_IN_PLACE, block_weights.data(), k, mpi::type::get<BlockWeight>(), MPI_SUM, MPI_COMM_WORLD
  );

  wrapper.p_graph = std::make_unique<DistributedPartitionedGraph>(
      wrapper.graph.get(), k, std::move(partition), std::move(block_weights)
  );

  const EdgeWeight cut = metrics::edge_cut(*wrapper.p_graph);
  const double imbalance = metrics::imbalance(*wrapper.p_graph);
  if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    std::cout << "INPUT cut=" << cut << " imbalance=" << imbalance << std::endl;
  }

  return wrapper;
}
} // namespace kaminpar::dist
