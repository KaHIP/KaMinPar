/*******************************************************************************
 * Sequential and parallel ParHiP parser.
 *
 * @file:   parhip_parser.cc
 * @author: Daniel Salwasser
 * @date:   15.02.2024
 ******************************************************************************/
#include "apps/io/shm_parhip_parser.h"

#include <cstddef>
#include <cstdint>
#include <functional>

#include <tbb/parallel_for.h>

#include "kaminpar-shm/graphutils/compressed_graph_builder.h"
#include "kaminpar-shm/graphutils/parallel_compressed_graph_builder.h"
#include "kaminpar-shm/graphutils/permutator.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

#include "apps/io/binary_util.h"

namespace {
using namespace kaminpar::io;

class ParHIPHeader {
  using NodeID = kaminpar::shm::NodeID;
  using EdgeID = kaminpar::shm::EdgeID;
  using NodeWeight = kaminpar::shm::NodeWeight;
  using EdgeWeight = kaminpar::shm::EdgeWeight;

public:
  static constexpr std::uint64_t kSize = 3 * sizeof(std::uint64_t);

  static ParHIPHeader parse(const BinaryReader &reader) {
    const auto version = reader.read<std::uint64_t>(0);
    const auto num_nodes = reader.read<std::uint64_t>(8);
    const auto num_edges = reader.read<std::uint64_t>(16);
    return ParHIPHeader(version, num_nodes, num_edges);
  }

  [[nodiscard]] static std::uint64_t version(
      const bool has_edge_weights,
      const bool has_node_weights,
      const bool has_64_bit_edge_id = sizeof(EdgeID) == 8,
      const bool has_64_bit_node_id = sizeof(NodeID) == 8,
      const bool has_64_bit_node_weight = sizeof(NodeWeight) == 8,
      const bool has_64_bit_edge_weight = sizeof(EdgeWeight) == 8
  ) {
    const auto make_flag = [&](const bool flag, const std::uint64_t shift) {
      return static_cast<std::uint64_t>(flag ? 0 : 1) << shift;
    };

    const std::uint64_t version =
        make_flag(has_64_bit_edge_weight, 5) | make_flag(has_64_bit_node_weight, 4) |
        make_flag(has_64_bit_node_id, 3) | make_flag(has_64_bit_edge_id, 2) |
        make_flag(has_node_weights, 1) | make_flag(has_edge_weights, 0);
    return version;
  }

  bool has_edge_weights;
  bool has_node_weights;
  bool has_64_bit_edge_id;
  bool has_64_bit_node_id;
  bool has_64_bit_node_weight;
  bool has_64_bit_edge_weight;
  std::uint64_t num_nodes;
  std::uint64_t num_edges;

  explicit ParHIPHeader(
      const std::uint64_t version, const std::uint64_t num_nodes, const std::uint64_t num_edges
  )
      : has_edge_weights((version & 1) == 0),
        has_node_weights((version & 2) == 0),
        has_64_bit_edge_id((version & 4) == 0),
        has_64_bit_node_id((version & 8) == 0),
        has_64_bit_node_weight((version & 16) == 0),
        has_64_bit_edge_weight((version & 32) == 0),
        num_nodes(num_nodes),
        num_edges(num_edges),
        _node_id_width(has_64_bit_node_id ? 8 : 4),
        _edge_id_width(has_64_bit_edge_id ? 8 : 4),
        _node_weight_width(has_64_bit_node_weight ? 8 : 4),
        _nodes_offset_base(ParHIPHeader::kSize + (num_nodes + 1) * _edge_id_width) {}

  [[nodiscard]] std::size_t nodes_offset() const {
    return ParHIPHeader::kSize;
  }

  [[nodiscard]] std::size_t edges_offset() const {
    return ParHIPHeader::kSize + (num_nodes + 1) * _edge_id_width;
  }

  [[nodiscard]] std::size_t node_weights_offset() const {
    return ParHIPHeader::kSize + (num_nodes + 1) * _edge_id_width + num_edges * _node_id_width;
  }

  [[nodiscard]] std::size_t edge_weights_offset() const {
    return ParHIPHeader::kSize + (num_nodes + 1) * _edge_id_width + num_edges * _node_id_width +
           (has_node_weights ? num_nodes * _node_weight_width : 0);
  }

  [[nodiscard]] EdgeID map_edge_offset(const EdgeID edge_offset) const {
    return (edge_offset - _nodes_offset_base) / _node_id_width;
  }

  void validate() const {
    if (has_64_bit_node_id && sizeof(NodeID) == 4) {
      LOG_ERROR << "The stored graph uses 64-Bit node IDs but this build uses 32-Bit node IDs.";
      std::exit(1);
    }

    if (has_64_bit_edge_id && sizeof(EdgeID) == 4) {
      LOG_ERROR << "The stored graph uses 64-Bit edge IDs but this build uses 32-Bit edge IDs.";
      std::exit(1);
    }

    if (has_node_weights && has_64_bit_node_weight && sizeof(NodeWeight) == 4) {
      LOG_ERROR
          << "The stored graph uses 64-Bit node weights but this build uses 32-Bit node weights.";
      std::exit(1);
    }

    if (has_edge_weights && has_64_bit_edge_weight && sizeof(EdgeWeight) == 4) {
      LOG_ERROR
          << "The stored graph uses 64-Bit edge weights but this build uses 32-Bit edge weights.";
      std::exit(1);
    }
  }

private:
  std::size_t _node_id_width;
  std::size_t _edge_id_width;
  std::size_t _node_weight_width;
  std::size_t _nodes_offset_base;
};

template <typename T, typename U = T, typename Transformer = std::identity>
kaminpar::StaticArray<T> read(
    const BinaryReader &reader,
    const std::size_t offset,
    const std::size_t length,
    Transformer transformer = {}
) {
  kaminpar::StaticArray<T> data(length, kaminpar::static_array::noinit);

  const U *raw_data = reader.fetch<U>(offset);
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, length), [&](const auto &r) {
    for (std::size_t i = r.begin(); i != r.end(); ++i) {
      data[i] = transformer(raw_data[i]);
    }
  });

  return data;
}

template <typename T, typename Transformer = std::identity>
kaminpar::StaticArray<T> read(
    const bool upcast,
    const BinaryReader &reader,
    const std::size_t offset,
    const std::size_t length,
    Transformer transformer = {}
) {
  if (upcast) {
    return read<T, std::uint32_t>(reader, offset, length, std::forward<Transformer>(transformer));
  } else {
    return read<T>(reader, offset, length, std::forward<Transformer>(transformer));
  }
}

} // namespace

namespace kaminpar::shm::io::parhip {
using namespace kaminpar::io;

CSRGraph csr_read(const std::string &filename, const bool sorted) {
  try {
    const BinaryReader reader(filename);
    const ParHIPHeader header = ParHIPHeader::parse(reader);
    header.validate();

    const bool upcast_edge_id = !header.has_64_bit_edge_id && sizeof(EdgeID) == 8;
    StaticArray<EdgeID> nodes = read<EdgeID>(
        upcast_edge_id,
        reader,
        header.nodes_offset(),
        header.num_nodes + 1,
        [&](const EdgeID e) { return header.map_edge_offset(e); }
    );

    const bool upcast_node_id = !header.has_64_bit_node_id && sizeof(NodeID) == 8;
    StaticArray<NodeID> edges =
        read<NodeID>(upcast_node_id, reader, header.edges_offset(), header.num_edges);

    StaticArray<NodeWeight> node_weights;
    if (header.has_node_weights) {
      const bool upcast_node_weight = !header.has_64_bit_node_weight && sizeof(NodeWeight) == 8;
      node_weights = read<NodeWeight>(
          upcast_node_weight, reader, header.node_weights_offset(), header.num_nodes
      );
    }

    StaticArray<EdgeWeight> edge_weights;
    if (header.has_edge_weights) {
      const bool upcast_edge_weight = !header.has_64_bit_edge_weight && sizeof(EdgeWeight) == 8;
      edge_weights = read<EdgeWeight>(
          upcast_edge_weight, reader, header.edge_weights_offset(), header.num_edges
      );
    }

    return CSRGraph(
        std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), sorted
    );
  } catch (const BinaryReaderException &e) {
    LOG_ERROR << e.what();
    std::exit(EXIT_FAILURE);
  }
}

CSRGraph csr_read_deg_buckets(const std::string &filename) {
  try {
    const BinaryReader reader(filename);
    const ParHIPHeader header = ParHIPHeader::parse(reader);
    header.validate();

    const auto *raw_nodes = reader.fetch<void>(header.nodes_offset());
    const bool upcast_edge_id = !header.has_64_bit_edge_id && sizeof(EdgeID) == 8;
    const auto node_mapper = [&](const NodeID u) -> EdgeID {
      if (upcast_edge_id) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_nodes)[u];
      } else {
        return reinterpret_cast<const EdgeID *>(raw_nodes)[u];
      }
    };

    const auto *raw_edges = reader.fetch<void>(header.edges_offset());
    const bool upcast_node_id = !header.has_64_bit_node_id && sizeof(NodeID) == 8;
    const auto edge_mapper = [&](const EdgeID e) -> NodeID {
      if (upcast_node_id) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_edges)[e];
      } else {
        return reinterpret_cast<const NodeID *>(raw_edges)[e];
      }
    };

    const auto *raw_node_weights = reader.fetch<void>(header.node_weights_offset());
    const bool upcast_node_weight = !header.has_64_bit_node_weight && sizeof(NodeWeight) == 8;
    const auto node_weight_mapper = [&](const NodeID u) -> NodeWeight {
      if (upcast_node_weight) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_node_weights)[u];
      } else {
        return reinterpret_cast<const NodeWeight *>(raw_node_weights)[u];
      }
    };

    const auto *raw_edge_weights = reader.fetch<void>(header.edge_weights_offset());
    const bool upcast_edge_weight = !header.has_64_bit_edge_weight && sizeof(EdgeWeight) == 8;
    const auto edge_weight_mapper = [&](const EdgeID e) -> EdgeWeight {
      if (upcast_edge_weight) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_edge_weights)[e];
      } else {
        return reinterpret_cast<const EdgeWeight *>(raw_edge_weights)[e];
      }
    };

    const auto degree = [&](const NodeID u) -> NodeID {
      return static_cast<NodeID>(
          header.map_edge_offset(node_mapper(u + 1)) - header.map_edge_offset(node_mapper(u))
      );
    };
    auto [perm, inv_perm] =
        graph::compute_node_permutation_by_degree_buckets(header.num_nodes, degree);

    StaticArray<EdgeID> nodes(header.num_nodes + 1, static_array::noinit);
    StaticArray<NodeWeight> node_weights;
    if (header.has_node_weights) {
      node_weights = StaticArray<NodeWeight>(header.num_nodes, static_array::noinit);
    }

    TIMED_SCOPE("Read nodes") {
      tbb::parallel_for(
          tbb::blocked_range<NodeID>(0, header.num_nodes),
          [&](const auto &local_nodes) {
            const NodeID local_nodes_start = local_nodes.begin();
            const NodeID local_nodes_end = local_nodes.end();

            for (NodeID u = local_nodes_start; u < local_nodes_end; ++u) {
              const NodeID old_u = inv_perm[u];

              nodes[u + 1] = degree(old_u);
              if (header.has_node_weights) [[unlikely]] {
                node_weights[u] = node_weight_mapper(old_u);
              }
            }
          }
      );
    };

    TIMED_SCOPE("Compute prefix sum") {
      nodes[0] = 0;
      parallel::prefix_sum(nodes.begin(), nodes.end(), nodes.begin());
    };

    StaticArray<NodeID> edges(header.num_edges, static_array::noinit);
    StaticArray<EdgeWeight> edge_weights;
    if (header.has_edge_weights) {
      edge_weights = StaticArray<EdgeWeight>(header.num_edges, static_array::noinit);
    }

    TIMED_SCOPE("Read edges") {
      tbb::parallel_for(
          tbb::blocked_range<NodeID>(0, header.num_nodes),
          [&](const auto &local_nodes) {
            const NodeID local_nodes_start = local_nodes.begin();
            const NodeID local_nodes_end = local_nodes.end();

            for (NodeID u = local_nodes_start; u < local_nodes_end; ++u) {
              const NodeID old_u = inv_perm[u];
              const EdgeID old_edge_start = header.map_edge_offset(node_mapper(old_u));
              const EdgeID old_edge_end = header.map_edge_offset(node_mapper(old_u + 1));

              EdgeID cur_edge = nodes[u];
              for (EdgeID old_edge = old_edge_start; old_edge < old_edge_end; ++old_edge) {
                const NodeID old_v = edge_mapper(old_edge);
                const NodeID v = perm[old_v];

                edges[cur_edge] = v;
                if (header.has_edge_weights) [[unlikely]] {
                  edge_weights[cur_edge] = edge_weight_mapper(old_edge);
                }

                cur_edge += 1;
              }
            }
          }
      );
    };

    CSRGraph csr_graph = CSRGraph(
        std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), true
    );

    csr_graph.set_permutation(std::move(perm));
    return csr_graph;
  } catch (const BinaryReaderException &e) {
    LOG_ERROR << e.what();
    std::exit(EXIT_FAILURE);
  }
}

CSRGraph csr_read(const std::string &filename, const NodeOrdering ordering) {
  if (ordering == NodeOrdering::EXTERNAL_DEGREE_BUCKETS) {
    return csr_read_deg_buckets(filename);
  }

  const bool sorted = ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS;
  return csr_read(filename, sorted);
}

CompressedGraph compressed_read(const std::string &filename, const bool sorted) {
  try {
    BinaryReader reader(filename);
    ParHIPHeader header = ParHIPHeader::parse(reader);
    header.validate();

    const auto *nodes = reader.fetch<void>(header.nodes_offset());
    const bool upcast_edge_id = !header.has_64_bit_edge_id && sizeof(EdgeID) == 8;
    const auto node = [&](const NodeID u) -> EdgeID {
      if (upcast_edge_id) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(nodes)[u];
      } else {
        return reinterpret_cast<const EdgeID *>(nodes)[u];
      }
    };

    const auto *edges = reader.fetch<void>(header.edges_offset());
    const bool upcast_node_id = !header.has_64_bit_node_id && sizeof(NodeID) == 8;
    const auto edge = [&](const EdgeID e) -> NodeID {
      if (upcast_node_id) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(edges)[e];
      } else {
        return reinterpret_cast<const NodeID *>(edges)[e];
      }
    };

    const auto *node_weights = reader.fetch<void>(header.node_weights_offset());
    const bool upcast_node_weight = !header.has_64_bit_node_weight && sizeof(NodeWeight) == 8;
    const auto node_weight = [&](const NodeID u) -> NodeWeight {
      if (upcast_node_weight) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(node_weights)[u];
      } else {
        return reinterpret_cast<const NodeWeight *>(node_weights)[u];
      }
    };

    const auto *edge_weights = reader.fetch<void>(header.edge_weights_offset());
    const bool upcast_edge_weight = !header.has_64_bit_edge_weight && sizeof(EdgeWeight) == 8;
    const auto edge_weight = [&](const EdgeID e) -> EdgeWeight {
      if (upcast_edge_weight) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(edge_weights)[e];
      } else {
        return reinterpret_cast<const EdgeWeight *>(edge_weights)[e];
      }
    };

    CompressedGraphBuilder builder(
        header.num_nodes, header.num_edges, header.has_node_weights, header.has_edge_weights, sorted
    );

    std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
    for (NodeID u = 0; u < header.num_nodes; ++u) {
      const EdgeID offset = header.map_edge_offset(node(u));
      const EdgeID next_offset = header.map_edge_offset(node(u + 1));

      const auto degree = static_cast<NodeID>(next_offset - offset);
      for (NodeID i = 0; i < degree; ++i) {
        const EdgeID e = offset + i;

        const NodeID adjacent_node = edge(e);
        const EdgeWeight weight = header.has_edge_weights ? edge_weight(e) : 1;

        neighbourhood.emplace_back(adjacent_node, weight);
      }

      builder.add_node(u, neighbourhood);
      if (header.has_node_weights) {
        builder.add_node_weight(u, node_weight(u));
      }

      neighbourhood.clear();
    }

    return builder.build();
  } catch (const BinaryReaderException &e) {
    LOG_ERROR << e.what();
    std::exit(EXIT_FAILURE);
  }
}

CompressedGraph compressed_read_parallel(const std::string &filename, const NodeOrdering ordering) {
  try {
    const BinaryReader reader(filename);
    const ParHIPHeader header = ParHIPHeader::parse(reader);
    header.validate();

    const auto *nodes = reader.fetch<void>(header.nodes_offset());
    const bool upcast_edge_id = !header.has_64_bit_edge_id && sizeof(EdgeID) == 8;
    const auto node = [&](const NodeID u) -> EdgeID {
      if (upcast_edge_id) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(nodes)[u];
      } else {
        return reinterpret_cast<const EdgeID *>(nodes)[u];
      }
    };

    const auto *edges = reader.fetch<void>(header.edges_offset());
    const bool upcast_node_id = !header.has_64_bit_node_id && sizeof(NodeID) == 8;
    const auto edge = [&](const EdgeID e) -> NodeID {
      if (upcast_node_id) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(edges)[e];
      } else {
        return reinterpret_cast<const NodeID *>(edges)[e];
      }
    };

    const auto *node_weights = reader.fetch<void>(header.node_weights_offset());
    const bool upcast_node_weight = !header.has_64_bit_node_weight && sizeof(NodeWeight) == 8;
    const auto node_weight = [&](const NodeID u) -> NodeWeight {
      if (upcast_node_weight) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(node_weights)[u];
      } else {
        return reinterpret_cast<const NodeWeight *>(node_weights)[u];
      }
    };

    const auto *edge_weights = reader.fetch<void>(header.edge_weights_offset());
    const bool upcast_edge_weight = !header.has_64_bit_edge_weight && sizeof(EdgeWeight) == 8;
    const auto edge_weight = [&](const EdgeID e) -> EdgeWeight {
      if (upcast_edge_weight) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(edge_weights)[e];
      } else {
        return reinterpret_cast<const EdgeWeight *>(edge_weights)[e];
      }
    };

    const bool sort_by_degree_bucket = ordering == NodeOrdering::EXTERNAL_DEGREE_BUCKETS;
    if (sort_by_degree_bucket) {
      const auto degree = [&](const NodeID u) -> NodeID {
        return static_cast<NodeID>(
            header.map_edge_offset(node(u + 1)) - header.map_edge_offset(node(u))
        );
      };

      auto [perm, inv_perm] =
          graph::compute_node_permutation_by_degree_buckets(header.num_nodes, degree);
      CompressedGraph compressed_graph = parallel_compress(
          header.num_nodes,
          header.num_edges,
          header.has_node_weights,
          header.has_edge_weights,
          true,
          [&](const NodeID u) { return inv_perm[u]; },
          degree,
          [&](const NodeID u) { return header.map_edge_offset(node(u)); },
          [&](const EdgeID e) { return perm[edge(e)]; },
          [&](const NodeID u) { return node_weight(u); },
          [&](const EdgeID e) { return edge_weight(e); }
      );

      compressed_graph.set_permutation(std::move(perm));
      return compressed_graph;
    } else {
      return parallel_compress(
          header.num_nodes,
          header.num_edges,
          header.has_node_weights,
          header.has_edge_weights,
          ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS,
          [](const NodeID u) { return u; },
          [&](const NodeID u) {
            return header.map_edge_offset(node(u + 1)) - header.map_edge_offset(node(u));
          },
          [&](const NodeID u) { return header.map_edge_offset(node(u)); },
          [&](const EdgeID e) { return edge(e); },
          [&](const NodeID u) { return node_weight(u); },
          [&](const EdgeID e) { return edge_weight(e); }
      );
    }
  } catch (const BinaryReaderException &e) {
    LOG_ERROR << e.what();
    std::exit(EXIT_FAILURE);
  }
}

void write(const std::string &filename, const CSRGraph &graph) {
  BinaryWriter writer(filename);

  const bool has_node_weights = graph.is_node_weighted();
  const bool has_edge_weights = graph.is_edge_weighted();

  const std::uint64_t version = ParHIPHeader::version(has_edge_weights, has_node_weights);
  writer.write_int(version);

  const std::uint64_t num_nodes = graph.n();
  writer.write_int(num_nodes);

  const std::uint64_t num_edges = graph.m();
  writer.write_int(num_edges);

  const NodeID num_total_nodes = num_nodes + 1;
  const EdgeID nodes_offset_base = ParHIPHeader::kSize + num_total_nodes * sizeof(EdgeID);
  const StaticArray<EdgeID> &nodes = graph.raw_nodes();

  StaticArray<EdgeID> raw_nodes(num_total_nodes, static_array::noinit);
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, num_total_nodes), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      raw_nodes[u] = nodes_offset_base + nodes[u] * sizeof(NodeID);
    }
  });

  writer.write_raw_static_array(raw_nodes);
  writer.write_raw_static_array(graph.raw_edges());

  if (has_node_weights) {
    writer.write_raw_static_array(graph.raw_node_weights());
  }

  if (has_edge_weights) {
    writer.write_raw_static_array(graph.raw_edge_weights());
  }
}

} // namespace kaminpar::shm::io::parhip
