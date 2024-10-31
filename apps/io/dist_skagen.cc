/*******************************************************************************
 * Utilities for graph generation with streaming KaGen.
 *
 * @file:   dist_skagen.h
 * @author: Daniel Salwasser
 * @date:   13.07.2024
 ******************************************************************************/
#include "apps/io/dist_skagen.h"

#include <limits>
#include <numeric>

#include <kagen.h>

#include "kaminpar-mpi/datatype.h"
#include "kaminpar-mpi/utils.h"

#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/logger.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/graph_compression/compressed_neighborhoods_builder.h"

namespace kaminpar::dist::io::skagen {

namespace {

SET_DEBUG(true);

}

DistributedCSRGraph csr_streaming_generate(
    const std::string &graph_options, const PEID chunks_per_pe, const MPI_Comm comm
) {
  const auto rank = mpi::get_comm_rank(comm);
  const auto size = mpi::get_comm_size(comm);

  LOG << "Generating graph " << graph_options << " with " << chunks_per_pe
      << " number of chunks per PE";

  ::kagen::sKaGen gen(graph_options, chunks_per_pe, MPI_COMM_WORLD);
  gen.Initialize();

  const auto range = gen.EstimateVertexRange();
  const auto first_node = range.first;
  const auto last_node = range.second;

  bool respects_esimated_vertex_range = true;

  for (std::size_t pe = 0; pe < static_cast<std::size_t>(size); ++pe) {
    LOG << "Vertices on PE " << std::setw(math::byte_width(pe)) << pe << ": "
        << gen.EstimateVertexRange(pe).first << " - " << gen.EstimateVertexRange(pe).second;
  }

  LLOG << "Generating ";

  StaticArray<GlobalNodeID> node_distribution(size + 1);
  node_distribution[rank + 1] = last_node;
  MPI_Allgather(
      MPI_IN_PLACE,
      0,
      MPI_DATATYPE_NULL,
      node_distribution.data() + 1,
      1,
      mpi::type::get<GlobalNodeID>(),
      comm
  );
  graph::GhostNodeMapper mapper(rank, node_distribution);

  const NodeID num_local_nodes = last_node - first_node;
  StaticArray<EdgeID> nodes(num_local_nodes + 1, static_array::noinit);

  std::size_t max_num_local_edges = num_local_nodes * node_distribution.back();
  if (max_num_local_edges / num_local_nodes != node_distribution.back()) {
    max_num_local_edges = std::numeric_limits<std::size_t>::max();
  }

  auto edges_ptr = heap_profiler::overcommit_memory<NodeID>(max_num_local_edges);
  auto edges = edges_ptr.get();

  NodeID visited_node = std::numeric_limits<NodeID>::max();
  NodeID current_node = 0;
  EdgeID current_edge = 0;
  while (gen.Continue()) {
    const kagen::StreamedGraph graph = gen.Next();

    graph.ForEachEdge([&](const kagen::SInt node, kagen::SInt adjacent_node) {
      if (visited_node != node) [[unlikely]] {
        if (visited_node == std::numeric_limits<NodeID>::max()) [[unlikely]] {
          visited_node = first_node;
        } else {
          visited_node += 1;
        }

        KASSERT(current_node < nodes.size());
        nodes[current_node++] = current_edge;

        while (visited_node < node) {
          visited_node += 1;

          KASSERT(current_node < nodes.size());
          nodes[current_node++] = current_edge;
        }
      }

      if (adjacent_node >= first_node && adjacent_node < last_node) {
        adjacent_node = adjacent_node - first_node;
      } else {
        adjacent_node = mapper.new_ghost_node(adjacent_node);
      }

      KASSERT(current_edge < max_num_local_edges);
      edges[current_edge++] = adjacent_node;

      if (node < first_node || node >= last_node) [[unlikely]] {
        respects_esimated_vertex_range = false;
      }
    });

    LLOG << ".";
  }

  if (num_local_nodes > 0) {
    while (visited_node < last_node) {
      visited_node += 1;
      nodes[current_node++] = current_edge;
    }
  }

  nodes[num_local_nodes] = current_edge;

  LOG;

  MPI_Allreduce(
      MPI_IN_PLACE, &respects_esimated_vertex_range, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD
  );
  if (!respects_esimated_vertex_range) {
    LOG << "Some edges on some PEs are out of the estimated vertex range!";
    std::exit(MPI_Finalize());
  }

  const EdgeID num_local_edges = current_edge;
  StaticArray<NodeID> wrapped_edges(num_local_edges, std::move(edges_ptr));
  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(edges, num_local_edges * sizeof(NodeID));
  }

  StaticArray<GlobalEdgeID> edge_distribution(size + 1);
  edge_distribution[rank] = num_local_edges;
  MPI_Allgather(
      MPI_IN_PLACE,
      1,
      mpi::type::get<GlobalEdgeID>(),
      edge_distribution.data(),
      1,
      mpi::type::get<GlobalEdgeID>(),
      comm
  );
  std::exclusive_scan(
      edge_distribution.begin(),
      edge_distribution.end(),
      edge_distribution.begin(),
      static_cast<GlobalEdgeID>(0)
  );

  auto [global_to_ghost, ghost_to_global, ghost_owner] = mapper.finalize();
  return DistributedCSRGraph(
      std::move(node_distribution),
      std::move(edge_distribution),
      std::move(nodes),
      std::move(wrapped_edges),
      std::move(ghost_owner),
      std::move(ghost_to_global),
      std::move(global_to_ghost),
      false,
      comm
  );
}

DistributedCompressedGraph compressed_streaming_generate(
    const std::string &graph_options, const PEID chunks_per_pe, const MPI_Comm comm
) {
  const auto rank = mpi::get_comm_rank(comm);
  const auto size = mpi::get_comm_size(comm);

  LOG << "Generating graph " << graph_options << " with " << chunks_per_pe
      << " number of chunks per PE";

  ::kagen::sKaGen gen(graph_options, chunks_per_pe, MPI_COMM_WORLD);
  gen.Initialize();

  const auto range = gen.EstimateVertexRange();
  const GlobalNodeID first_node = range.first;
  const GlobalNodeID last_node = range.second;

  bool respects_esimated_vertex_range = true;

  for (std::size_t pe = 0; pe < static_cast<std::size_t>(size); ++pe) {
    LOG << "Vertices on PE " << std::setw(math::byte_width(pe)) << pe << ": "
        << gen.EstimateVertexRange(pe).first << " - " << gen.EstimateVertexRange(pe).second;
  }

  LLOG << "Generating ";

  StaticArray<GlobalNodeID> node_distribution(size + 1);
  node_distribution[rank + 1] = last_node;
  MPI_Allgather(
      MPI_IN_PLACE,
      0,
      MPI_DATATYPE_NULL,
      node_distribution.data() + 1,
      1,
      mpi::type::get<GlobalNodeID>(),
      comm
  );
  CompactGhostNodeMappingBuilder mapper(rank, node_distribution);

  const NodeID num_local_nodes = static_cast<NodeID>(last_node - first_node);
  std::size_t max_num_local_edges = num_local_nodes * node_distribution.back();
  if (max_num_local_edges / num_local_nodes != node_distribution.back()) {
    max_num_local_edges = std::numeric_limits<std::size_t>::max();
  }

  CompressedNeighborhoodsBuilder<NodeID, EdgeID, EdgeWeight> builder(
      num_local_nodes, max_num_local_edges, false
  );

  EdgeID num_local_edges = 0;
  GlobalNodeID current_node = std::numeric_limits<GlobalNodeID>::max();

  std::vector<NodeID> neighbourhood;
  std::vector<NodeID> empty_neighbourhood(0);

  const auto compress_neighborhood = [&] {
    num_local_edges += neighbourhood.size();
    builder.add(current_node - first_node, neighbourhood);
    neighbourhood.clear();
  };

  const auto add_isolated_node = [&] {
    builder.add(current_node - first_node, empty_neighbourhood);
  };

  while (gen.Continue()) {
    const kagen::StreamedGraph graph = gen.Next();

    graph.ForEachEdge([&](const kagen::SInt node, kagen::SInt adjacent_node) {
      if (current_node != node) [[unlikely]] {
        if (current_node == std::numeric_limits<GlobalNodeID>::max()) [[unlikely]] {
          current_node = first_node;

          while (current_node < node) {
            add_isolated_node();
            current_node++;
          }
        } else {
          compress_neighborhood();
          current_node++;

          while (current_node < node) {
            add_isolated_node();
            current_node++;
          }
        }
      }

      if (adjacent_node >= first_node && adjacent_node < last_node) {
        adjacent_node = adjacent_node - first_node;
      } else {
        adjacent_node = mapper.new_ghost_node(adjacent_node);
      }

      neighbourhood.push_back(adjacent_node);

      if (node < first_node || node >= last_node) [[unlikely]] {
        respects_esimated_vertex_range = false;
      }
    });

    LLOG << ".";
  }

  if (num_local_nodes > 0) {
    compress_neighborhood();
    current_node++;

    while (current_node < last_node) {
      add_isolated_node();
      current_node++;
    }
  }

  LOG;

  MPI_Allreduce(
      MPI_IN_PLACE, &respects_esimated_vertex_range, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD
  );
  if (!respects_esimated_vertex_range) {
    LOG << "Some edges on some PEs are out of the estimated vertex range!";
    std::exit(MPI_Finalize());
  }

  StaticArray<GlobalEdgeID> edge_distribution(size + 1);
  edge_distribution[rank] = num_local_edges;
  MPI_Allgather(
      MPI_IN_PLACE,
      1,
      mpi::type::get<GlobalEdgeID>(),
      edge_distribution.data(),
      1,
      mpi::type::get<GlobalEdgeID>(),
      comm
  );
  std::exclusive_scan(
      edge_distribution.begin(),
      edge_distribution.end(),
      edge_distribution.begin(),
      static_cast<GlobalEdgeID>(0)
  );

  builder.set_num_edges(num_local_edges);
  return DistributedCompressedGraph(
      std::move(node_distribution),
      std::move(edge_distribution),
      builder.build(),
      {},
      mapper.finalize(),
      false,
      comm
  );
}

} // namespace kaminpar::dist::io::skagen
