/*******************************************************************************
 * IO utilities for the shared-memory partitioner.
 *
 * @file:   kaminpar-io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-io/kaminpar_io.h"

#include <fstream>
#include <numeric>
#include <optional>

#include "kaminpar-io/graph_compression_binary.h"
#include "kaminpar-io/metis_parser.h"
#include "kaminpar-io/parhip_parser.h"
#include "kaminpar-io/util/file_toker.h"

namespace kaminpar::shm::io {

std::optional<Graph> read_graph(
    const std::string &filename,
    const GraphFileFormat file_format,
    const bool compress,
    const NodeOrdering ordering
) {
  switch (file_format) {
  case GraphFileFormat::METIS:
    return metis::read_graph(filename, compress, ordering);
  case GraphFileFormat::PARHIP:
    return parhip::read_graph(filename, compress, ordering);
    break;
  case GraphFileFormat::COMPRESSED:
    return compressed_binary::read(filename);
  default:
    return std::nullopt;
  }
}

void write_graph(
    const std::string &filename, const GraphFileFormat file_format, const Graph &graph
) {
  switch (file_format) {
  case GraphFileFormat::METIS:
    metis::write_graph(filename, graph);
    break;
  case GraphFileFormat::PARHIP:
    parhip::write_graph(filename, graph);
    break;
  case GraphFileFormat::COMPRESSED:
    if (graph.is_compressed()) {
      compressed_binary::write(filename, graph.compressed_graph());
    }
    break;
  }
}

void write_partition(const std::string &filename, const std::span<const BlockID> partition) {
  std::ofstream out(filename);
  for (const BlockID block : partition) {
    out << block << "\n";
  }
}

std::vector<BlockID> read_partition(const std::string &filename) {
  using namespace kaminpar::io;
  MappedFileToker toker(filename);

  std::vector<BlockID> partition;
  while (toker.valid_position()) {
    partition.push_back(toker.scan_uint());
    toker.consume_char('\n');
  }

  return {partition.begin(), partition.end()};
}

void write_block_sizes(
    const std::string &filename, const BlockID k, const std::span<const BlockID> partition
) {
  std::vector<NodeID> block_sizes(k, 0);
  for (const BlockID block : partition) {
    block_sizes[block]++;
  }

  std::ofstream out(filename);
  for (const BlockID block_size : block_sizes) {
    out << block_size << "\n";
  }
}

std::vector<BlockID> read_block_sizes(const std::string &filename) {
  using namespace kaminpar::io;
  MappedFileToker toker(filename);

  std::vector<NodeID> block_sizes;
  while (toker.valid_position()) {
    block_sizes.push_back(toker.scan_uint());
    toker.consume_char('\n');
  }

  const NodeID n = std::accumulate(block_sizes.begin(), block_sizes.end(), 0);

  std::vector<BlockID> partition(n);
  NodeID cur = 0;
  BlockID block = 0;

  for (const NodeID block_size : block_sizes) {
    std::fill(partition.begin() + cur, partition.begin() + cur + block_size, block);
    cur += block_size;
    ++block;
  }

  return partition;
}

void write_remapping(const std::string &filename, std::span<const NodeID> remapping) {
  std::ofstream out(filename);
  for (const NodeID new_id : remapping) {
    out << new_id << "\n";
  }
}

} // namespace kaminpar::shm::io
