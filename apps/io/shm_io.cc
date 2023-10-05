/*******************************************************************************
 * IO utilities for the shared-memory partitioner.
 *
 * @file:   shm_io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "apps/io/shm_io.h"

#include <fstream>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

#include "apps/io/metis_parser.h"

namespace kaminpar::shm::io {
//
// Public Metis functions
//
namespace metis {
template <bool checked>
void read(
    const std::string &filename,
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
) {
  using namespace kaminpar::io::metis;

  bool store_node_weights = false;
  bool store_edge_weights = false;
  std::int64_t total_node_weight = 0;
  std::int64_t total_edge_weight = 0;

  NodeID u = 0;
  EdgeID e = 0;

  parse<false>(
      filename,
      [&](const auto &format) {
        if constexpr (checked) {
          if (format.number_of_nodes >=
              static_cast<std::uint64_t>(std::numeric_limits<NodeID>::max())) {
            LOG_ERROR << "number of nodes is too large for the node ID type";
            std::exit(1);
          }
          if (format.number_of_edges >=
              static_cast<std::uint64_t>(std::numeric_limits<EdgeID>::max())) {
            LOG_ERROR << "number of edges is too large for the edge ID type";
            std::exit(1);
          }
          if (format.number_of_edges >
              (format.number_of_nodes * (format.number_of_nodes - 1) / 2)) {
            LOG_ERROR << "specified number of edges is impossibly large";
            std::exit(1);
          }
        } else {
          KASSERT(
              format.number_of_nodes <=
                  static_cast<std::uint64_t>(std::numeric_limits<NodeID>::max()),
              "number of nodes is too large for the node ID type"
          );
          KASSERT(
              format.number_of_edges <=
                  static_cast<std::uint64_t>(std::numeric_limits<EdgeID>::max()),
              "number of edges is too large for the edge ID type"
          );
          KASSERT(
              format.number_of_edges <= (format.number_of_nodes * (format.number_of_nodes - 1)) / 2,
              "specified number of edges is impossibly large"
          );
        }

        store_node_weights = format.has_node_weights;
        store_edge_weights = format.has_edge_weights;
        nodes.resize(format.number_of_nodes + 1);
        edges.resize(format.number_of_edges * 2);
        if (store_node_weights) {
          node_weights.resize(format.number_of_nodes);
        }
        if (store_edge_weights) {
          edge_weights.resize(format.number_of_edges * 2);
        }
      },
      [&](const std::uint64_t weight) {
        if constexpr (checked) {
          if (weight > static_cast<std::uint64_t>(std::numeric_limits<NodeWeight>::max())) {
            LOG_ERROR << "node weight is too large for the node weight type";
            std::exit(1);
          }
          if (weight <= 0) {
            LOG_ERROR << "zero node weights are not supported";
            std::exit(1);
          }
        } else {
          KASSERT(
              weight <= static_cast<std::uint64_t>(std::numeric_limits<NodeWeight>::max()),
              "node weight is too large for the node weight type"
          );
          KASSERT(weight > 0u, "zero node weights are not supported");
        }

        if (store_node_weights) {
          node_weights[u] = static_cast<NodeWeight>(weight);
        }
        nodes[u] = e;
        total_node_weight += weight;
        ++u;
      },
      [&](const std::uint64_t weight, const std::uint64_t v) {
        if constexpr (checked) {
          if (weight > static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max())) {
            LOG_ERROR << "edge weight is too large for the edge weight type";
            std::exit(1);
          }
          if (weight <= 0) {
            LOG_ERROR << "zero edge weights are not supported";
            std::exit(1);
          }
          if (v + 1 >= nodes.size()) {
            LOG_ERROR << "neighbor " << v + 1 << " of nodes " << u + 1 << " is out of bounds";
            std::exit(1);
          }
          if (v + 1 == u) {
            LOG_ERROR << "detected self-loop on node " << v + 1 << ", which is not allowed";
            std::exit(1);
          }
        } else {
          KASSERT(
              weight <= static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max()),
              "edge weight is too large for the edge weight type"
          );
          KASSERT(weight > 0u, "zero edge weights are not supported");
          KASSERT(v + 1 < nodes.size(), "neighbor out of bounds");
          KASSERT(u != v + 1, "detected illegal self-loop");
        }

        if (store_edge_weights) {
          edge_weights[e] = static_cast<EdgeWeight>(weight);
        }
        edges[e] = static_cast<NodeID>(v);
        total_edge_weight += weight;
        ++e;
      }
  );
  nodes[u] = e;

  if constexpr (checked) {
    if (total_node_weight > static_cast<std::int64_t>(std::numeric_limits<NodeWeight>::max())) {
      LOG_ERROR << "total node weight does not fit into the node weight type";
      std::exit(1);
    }
    if (total_edge_weight > static_cast<std::int64_t>(std::numeric_limits<EdgeWeight>::max())) {
      LOG_ERROR << "total edge weight does not fit into the edge weight type";
      std::exit(1);
    }
  } else {
    KASSERT(
        total_node_weight <= static_cast<std::int64_t>(std::numeric_limits<NodeWeight>::max()),
        "total node weight does not fit into the node weight type"
    );
    KASSERT(
        total_edge_weight <= static_cast<std::int64_t>(std::numeric_limits<EdgeWeight>::max()),
        "total edge weight does not fit into the edge weight type"
    );
  }

  // only keep weights if the graph is really weighted
  const bool unit_node_weights = static_cast<NodeID>(total_node_weight + 1) == nodes.size();
  if (unit_node_weights) {
    node_weights.free();
  }

  const bool unit_edge_weights = static_cast<EdgeID>(total_edge_weight) == edges.size();
  if (unit_edge_weights) {
    edge_weights.free();
  }
}

template void read<false>(
    const std::string &filename,
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
);

template void read<true>(
    const std::string &filename,
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
);
} // namespace metis

//
// Partition
//

namespace partition {
void write(const std::string &filename, const std::vector<BlockID> &partition) {
  std::ofstream out(filename);
  for (const BlockID block : partition) {
    out << block << "\n";
  }
}

std::vector<BlockID> read(const std::string &filename) {
  using namespace kaminpar::io;

  MappedFileToker<> toker(filename);
  std::vector<BlockID> partition;
  while (toker.valid_position()) {
    partition.push_back(toker.scan_uint());
    toker.consume_char('\n');
  }

  return partition;
}
} // namespace partition
} // namespace kaminpar::shm::io
