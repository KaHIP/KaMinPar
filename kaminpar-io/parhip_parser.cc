/*******************************************************************************
 * Sequential and parallel ParHiP parser.
 *
 * @file:   parhip_parser.cc
 * @author: Daniel Salwasser
 * @date:   15.02.2024
 ******************************************************************************/
#include "kaminpar-io/parhip_parser.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-io/util/binary_util.h"

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/graphutils/compressed_graph_builder.h"
#include "kaminpar-shm/graphutils/permutator.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::io::parhip {

using namespace kaminpar::io;

namespace {

class ParHIPHeader {
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
StaticArray<T> read(
    const BinaryReader &reader,
    const std::size_t offset,
    const std::size_t length,
    Transformer transformer = {}
) {
  StaticArray<T> data(length, static_array::noinit);

  const U *raw_data = reader.fetch<U>(offset);
  tbb::parallel_for<std::size_t>(0, length, [&](const auto i) {
    data[i] = transformer(raw_data[i]);
  });

  return data;
}

template <typename T, typename Transformer = std::identity>
StaticArray<T> read(
    const BinaryReader &reader,
    const std::size_t offset,
    const std::size_t length,
    const bool upcast,
    Transformer transformer = {}
) {
  if (upcast) {
    return read<T, std::uint32_t>(reader, offset, length, std::forward<Transformer>(transformer));
  } else {
    return read<T>(reader, offset, length, std::forward<Transformer>(transformer));
  }
}

std::optional<CSRGraph> csr_read(const std::string &filename, const bool sorted) {
  try {
    const BinaryReader reader(filename);
    const ParHIPHeader header = ParHIPHeader::parse(reader);
    header.validate();

    const bool upcast_edge_id = !header.has_64_bit_edge_id && sizeof(EdgeID) == 8;
    StaticArray<EdgeID> nodes = read<EdgeID>(
        reader,
        header.nodes_offset(),
        header.num_nodes + 1,
        upcast_edge_id,
        [&](const EdgeID e) { return header.map_edge_offset(e); }
    );

    const bool upcast_node_id = !header.has_64_bit_node_id && sizeof(NodeID) == 8;
    StaticArray<NodeID> edges =
        read<NodeID>(reader, header.edges_offset(), header.num_edges, upcast_node_id);

    StaticArray<NodeWeight> node_weights;
    if (header.has_node_weights) {
      const bool upcast_node_weight = !header.has_64_bit_node_weight && sizeof(NodeWeight) == 8;
      node_weights = read<NodeWeight>(
          reader, header.node_weights_offset(), header.num_nodes, upcast_node_weight
      );
    }

    StaticArray<EdgeWeight> edge_weights;
    if (header.has_edge_weights) {
      const bool upcast_edge_weight = !header.has_64_bit_edge_weight && sizeof(EdgeWeight) == 8;
      edge_weights = read<EdgeWeight>(
          reader, header.edge_weights_offset(), header.num_edges, upcast_edge_weight
      );
    }

    return CSRGraph(
        std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), sorted
    );
  } catch (const BinaryReaderException &e) {
    return std::nullopt;
  }
}

std::optional<CSRGraph> csr_read_deg_buckets(const std::string &filename) {
  try {
    const BinaryReader reader(filename);
    const ParHIPHeader header = ParHIPHeader::parse(reader);
    header.validate();

    const auto *raw_nodes = reader.fetch<void>(header.nodes_offset());
    const bool upcast_edge_id = !header.has_64_bit_edge_id && sizeof(EdgeID) == 8;
    const auto fetch_edge_offset = [&](const NodeID u) -> EdgeID {
      if (upcast_edge_id) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_nodes)[u];
      } else {
        return reinterpret_cast<const EdgeID *>(raw_nodes)[u];
      }
    };

    const auto *raw_edges = reader.fetch<void>(header.edges_offset());
    const bool upcast_node_id = !header.has_64_bit_node_id && sizeof(NodeID) == 8;
    const auto fetch_adjacent_node = [&](const EdgeID e) -> NodeID {
      if (upcast_node_id) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_edges)[e];
      } else {
        return reinterpret_cast<const NodeID *>(raw_edges)[e];
      }
    };

    const auto *raw_node_weights = reader.fetch<void>(header.node_weights_offset());
    const bool upcast_node_weight = !header.has_64_bit_node_weight && sizeof(NodeWeight) == 8;
    const auto fetch_node_weight = [&](const NodeID u) -> NodeWeight {
      if (upcast_node_weight) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_node_weights)[u];
      } else {
        return reinterpret_cast<const NodeWeight *>(raw_node_weights)[u];
      }
    };

    const auto *raw_edge_weights = reader.fetch<void>(header.edge_weights_offset());
    const bool upcast_edge_weight = !header.has_64_bit_edge_weight && sizeof(EdgeWeight) == 8;
    const auto fetch_edge_weight = [&](const EdgeID e) -> EdgeWeight {
      if (upcast_edge_weight) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_edge_weights)[e];
      } else {
        return reinterpret_cast<const EdgeWeight *>(raw_edge_weights)[e];
      }
    };

    const auto fetch_degree = [&](const NodeID u) -> NodeID {
      return static_cast<NodeID>(
          header.map_edge_offset(fetch_edge_offset(u + 1)) -
          header.map_edge_offset(fetch_edge_offset(u))
      );
    };

    auto [perm, inv_perm] =
        graph::compute_node_permutation_by_degree_buckets(header.num_nodes, fetch_degree);

    StaticArray<EdgeID> nodes(header.num_nodes + 1, static_array::noinit);
    StaticArray<NodeWeight> node_weights;
    if (header.has_node_weights) {
      node_weights = StaticArray<NodeWeight>(header.num_nodes, static_array::noinit);
    }

    TIMED_SCOPE("Read nodes") {
      tbb::parallel_for<NodeID>(0, header.num_nodes, [&](const NodeID old_u) {
        const NodeID new_u = perm[old_u];

        nodes[new_u + 1] = fetch_degree(old_u);
        if (header.has_node_weights) [[unlikely]] {
          node_weights[new_u] = fetch_node_weight(old_u);
        }
      });
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
      tbb::parallel_for<NodeID>(0, header.num_nodes, [&](const NodeID old_u) {
        const NodeID new_u = perm[old_u];
        const EdgeID old_edge_start = header.map_edge_offset(fetch_edge_offset(old_u));
        const EdgeID old_edge_end = header.map_edge_offset(fetch_edge_offset(old_u + 1));

        EdgeID cur_edge = nodes[new_u];
        for (EdgeID old_edge = old_edge_start; old_edge < old_edge_end; ++old_edge) {
          const NodeID old_v = fetch_adjacent_node(old_edge);
          const NodeID v = perm[old_v];

          edges[cur_edge] = v;
          if (header.has_edge_weights) [[unlikely]] {
            edge_weights[cur_edge] = fetch_edge_weight(old_edge);
          }

          cur_edge += 1;
        }
      });
    };

    CSRGraph csr_graph = CSRGraph(
        std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), true
    );
    csr_graph.set_permutation(std::move(perm));

    return csr_graph;
  } catch (const BinaryReaderException &e) {
    return std::nullopt;
  }
}

} // namespace

std::optional<CSRGraph> csr_read(const std::string &filename, const NodeOrdering ordering) {
  if (ordering == NodeOrdering::EXTERNAL_DEGREE_BUCKETS) {
    return csr_read_deg_buckets(filename);
  }

  const bool sorted = ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS;
  return csr_read(filename, sorted);
}

namespace {

std::optional<CompressedGraph> compressed_read(const std::string &filename, const bool sorted) {
  try {
    const BinaryReader reader(filename);
    const ParHIPHeader header = ParHIPHeader::parse(reader);
    header.validate();

    const bool upcast_edge_id = !header.has_64_bit_edge_id && sizeof(EdgeID) == 8;
    StaticArray<EdgeID> nodes = read<EdgeID>(
        reader,
        header.nodes_offset(),
        header.num_nodes + 1,
        upcast_edge_id,
        [&](const EdgeID e) { return header.map_edge_offset(e); }
    );

    StaticArray<NodeWeight> node_weights;
    if (header.has_node_weights) {
      const bool upcast_node_weight = !header.has_64_bit_node_weight && sizeof(NodeWeight) == 8;
      node_weights = read<NodeWeight>(
          reader, header.node_weights_offset(), header.num_nodes, upcast_node_weight
      );
    }

    const auto fetch_degree = [&](const NodeID u) {
      return static_cast<NodeID>(nodes[u + 1] - nodes[u]);
    };

    const auto *edges = reader.fetch<void>(header.edges_offset());
    const bool upcast_node_id = !header.has_64_bit_node_id && sizeof(NodeID) == 8;
    const auto fetch_adjacent_node = [&](const EdgeID e) -> NodeID {
      if (upcast_node_id) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(edges)[e];
      } else {
        return reinterpret_cast<const NodeID *>(edges)[e];
      }
    };

    const auto *edge_weights = reader.fetch<void>(header.edge_weights_offset());
    const bool upcast_edge_weight = !header.has_64_bit_edge_weight && sizeof(EdgeWeight) == 8;
    const auto fetch_edge_weight = [&](const EdgeID e) -> EdgeWeight {
      if (upcast_edge_weight) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(edge_weights)[e];
      } else {
        return reinterpret_cast<const EdgeWeight *>(edge_weights)[e];
      }
    };

    if (header.has_edge_weights) {
      using Edge = std::pair<NodeID, EdgeWeight>;
      const auto fetch_neighborhood = [&](const NodeID u, std::span<Edge> neighborhood) {
        const EdgeID first_edge = nodes[u];
        const EdgeID last_edge = nodes[u + 1];
        const NodeID degree = static_cast<NodeID>(last_edge - first_edge);

        for (NodeID i = 0; i < degree; ++i) {
          const EdgeID e = first_edge + i;
          neighborhood[i] = std::make_pair(fetch_adjacent_node(e), fetch_edge_weight(e));
        }
      };

      return parallel_compress_weighted(
          header.num_nodes,
          header.num_edges,
          fetch_degree,
          fetch_neighborhood,
          std::move(node_weights),
          sorted
      );
    } else {
      const auto fetch_neighborhood = [&](const NodeID u, std::span<NodeID> neighborhood) {
        const EdgeID first_edge = nodes[u];
        const EdgeID last_edge = nodes[u + 1];
        const NodeID degree = static_cast<NodeID>(last_edge - first_edge);

        for (NodeID i = 0; i < degree; ++i) {
          const EdgeID e = first_edge + i;
          neighborhood[i] = fetch_adjacent_node(e);
        }
      };

      return parallel_compress(
          header.num_nodes,
          header.num_edges,
          fetch_degree,
          fetch_neighborhood,
          std::move(node_weights),
          sorted
      );
    }
  } catch (const BinaryReaderException &e) {
    return std::nullopt;
  }
}

std::optional<CompressedGraph> compressed_read_deg_buckets(const std::string &filename) {
  try {
    const BinaryReader reader(filename);
    const ParHIPHeader header = ParHIPHeader::parse(reader);
    header.validate();

    const auto *raw_nodes = reader.fetch<void>(header.nodes_offset());
    const bool upcast_edge_id = !header.has_64_bit_edge_id && sizeof(EdgeID) == 8;
    const auto fetch_edge_offset = [&](const NodeID u) -> EdgeID {
      if (upcast_edge_id) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_nodes)[u];
      } else {
        return reinterpret_cast<const EdgeID *>(raw_nodes)[u];
      }
    };

    const auto *raw_edges = reader.fetch<void>(header.edges_offset());
    const bool upcast_node_id = !header.has_64_bit_node_id && sizeof(NodeID) == 8;
    const auto fetch_adjacent_node = [&](const EdgeID e) -> NodeID {
      if (upcast_node_id) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_edges)[e];
      } else {
        return reinterpret_cast<const NodeID *>(raw_edges)[e];
      }
    };

    const auto *raw_node_weights = reader.fetch<void>(header.node_weights_offset());
    const bool upcast_node_weight = !header.has_64_bit_node_weight && sizeof(NodeWeight) == 8;
    const auto fetch_node_weight = [&](const NodeID u) -> NodeWeight {
      if (upcast_node_weight) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_node_weights)[u];
      } else {
        return reinterpret_cast<const NodeWeight *>(raw_node_weights)[u];
      }
    };

    const auto *raw_edge_weights = reader.fetch<void>(header.edge_weights_offset());
    const bool upcast_edge_weight = !header.has_64_bit_edge_weight && sizeof(EdgeWeight) == 8;
    const auto fetch_edge_weight = [&](const EdgeID e) -> EdgeWeight {
      if (upcast_edge_weight) [[unlikely]] {
        return reinterpret_cast<const std::uint32_t *>(raw_edge_weights)[e];
      } else {
        return reinterpret_cast<const EdgeWeight *>(raw_edge_weights)[e];
      }
    };

    const auto fetch_degree = [&](const NodeID u) -> NodeID {
      return static_cast<NodeID>(
          header.map_edge_offset(fetch_edge_offset(u + 1)) -
          header.map_edge_offset(fetch_edge_offset(u))
      );
    };

    auto [perm, inv_perm] =
        graph::compute_node_permutation_by_degree_buckets(header.num_nodes, fetch_degree);

    ParallelCompressedGraphBuilder builder(
        header.num_nodes, header.num_edges, header.has_node_weights, header.has_edge_weights, true
    );

    if (header.has_node_weights) {
      SCOPED_TIMER("Read node weights");

      tbb::parallel_for<NodeID>(0, header.num_nodes, [&](const NodeID old_u) {
        const NodeID new_u = perm[old_u];
        builder.add_node_weight(new_u, fetch_node_weight(old_u));
      });
    }

    tbb::enumerable_thread_specific<std::vector<std::pair<NodeID, EdgeWeight>>> neighbourhood_ets;

    TIMED_SCOPE("First pass") {
      tbb::parallel_for<NodeID>(0, header.num_nodes, [&](const NodeID old_u) {
        auto &neighborhood = neighbourhood_ets.local();

        const NodeID new_u = perm[old_u];
        const EdgeID old_edge_start = header.map_edge_offset(fetch_edge_offset(old_u));
        const EdgeID old_edge_end = header.map_edge_offset(fetch_edge_offset(old_u + 1));

        for (EdgeID old_edge = old_edge_start; old_edge < old_edge_end; ++old_edge) {
          const NodeID old_v = fetch_adjacent_node(old_edge);
          const NodeID v = perm[old_v];
          const EdgeWeight w = header.has_edge_weights ? fetch_edge_weight(old_edge) : 1;
          neighborhood.emplace_back(v, w);
        }

        builder.register_neighborhood(new_u, neighborhood);
        neighborhood.clear();
      });
    };

    TIMED_SCOPE("Compute offsets") {
      builder.compute_offsets();
    };

    TIMED_SCOPE("Second pass") {
      tbb::parallel_for<NodeID>(0, header.num_nodes, [&](const NodeID old_u) {
        auto &neighborhood = neighbourhood_ets.local();

        const NodeID new_u = perm[old_u];
        const EdgeID old_edge_start = header.map_edge_offset(fetch_edge_offset(old_u));
        const EdgeID old_edge_end = header.map_edge_offset(fetch_edge_offset(old_u + 1));

        for (EdgeID old_edge = old_edge_start; old_edge < old_edge_end; ++old_edge) {
          const NodeID old_v = fetch_adjacent_node(old_edge);
          const NodeID v = perm[old_v];
          const EdgeWeight w = header.has_edge_weights ? fetch_edge_weight(old_edge) : 1;
          neighborhood.emplace_back(v, w);
        }

        builder.add_neighborhood(new_u, neighborhood);
        neighborhood.clear();
      });
    };

    CompressedGraph compressed_graph = builder.build();
    compressed_graph.set_permutation(std::move(perm));

    return compressed_graph;
  } catch (const BinaryReaderException &e) {
    return std::nullopt;
  }
}

} // namespace

std::optional<CompressedGraph>
compressed_read(const std::string &filename, const NodeOrdering ordering) {
  if (ordering == NodeOrdering::EXTERNAL_DEGREE_BUCKETS) {
    return compressed_read_deg_buckets(filename);
  }

  const bool sorted = ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS;
  return compressed_read(filename, sorted);
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
  tbb::parallel_for<NodeID>(0, num_total_nodes, [&](const NodeID u) {
    raw_nodes[u] = nodes_offset_base + nodes[u] * sizeof(NodeID);
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
