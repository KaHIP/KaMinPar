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

#include "kaminpar-common/logger.h"

namespace kaminpar::shm::io {

std::unordered_map<std::string, GraphFileFormat> get_graph_file_formats() {
  return {
      {"metis", GraphFileFormat::METIS},
      {"parhip", GraphFileFormat::PARHIP},
  };
}

std::optional<CSRGraph> csr_read(
    const std::string &filename, const GraphFileFormat file_format, const NodeOrdering ordering
) {
  switch (file_format) {
  case GraphFileFormat::METIS: {
    if (ordering == NodeOrdering::EXTERNAL_DEGREE_BUCKETS) {
      LOG_WARNING << "A graph stored in METIS format cannot be rearranged by degree buckets during "
                     "IO.";
    }

    const bool sorted = ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS;
    return metis::csr_read(filename, sorted);
  }
  case GraphFileFormat::PARHIP:
    return parhip::csr_read(filename, ordering);
  default:
    return std::nullopt;
  }
}

std::optional<CompressedGraph> compressed_read(
    const std::string &filename, const GraphFileFormat file_format, const NodeOrdering ordering
) {
  switch (file_format) {
  case GraphFileFormat::METIS: {
    if (ordering == NodeOrdering::EXTERNAL_DEGREE_BUCKETS) {
      LOG_WARNING << "A graph stored in METIS format cannot be rearranged by degree buckets during "
                     "IO.";
    }

    const bool sorted = ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS;
    return metis::compress_read(filename, sorted);
  }
  case GraphFileFormat::PARHIP:
    return parhip::compressed_read(filename, ordering);
  default:
    return std::nullopt;
  }
}

std::optional<Graph> read(
    const std::string &filename,
    const GraphFileFormat file_format,
    const NodeOrdering ordering,
    const bool compress
) {
  if (compress) {
    if (compressed_binary::is_compressed(filename)) {
      if (auto compressed_graph = compressed_binary::read(filename)) {
        return Graph(std::make_unique<CompressedGraph>(std::move(*compressed_graph)));
      }
    } else {
      if (auto compressed_graph = compressed_read(filename, file_format, ordering)) {
        return Graph(std::make_unique<CompressedGraph>(std::move(*compressed_graph)));
      }
    }
  }

  if (auto csr_graph = csr_read(filename, file_format, ordering)) {
    return Graph(std::make_unique<CSRGraph>(std::move(*csr_graph)));
  }

  return std::nullopt;
}

namespace partition {

void write(const std::string &filename, const std::span<const BlockID> partition) {
  std::ofstream out(filename);
  for (const BlockID block : partition) {
    out << block << "\n";
  }
}

StaticArray<BlockID> read(const std::string &filename) {
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
  std::vector<NodeID> block_sizes(k);
  for (const BlockID block : partition) {
    block_sizes[block]++;
  }

  std::ofstream out(filename);
  for (const BlockID block_size : block_sizes) {
    out << block_size << "\n";
  }
}

StaticArray<BlockID> read_block_sizes(const std::string &filename) {
  using namespace kaminpar::io;
  MappedFileToker toker(filename);

  std::vector<NodeID> block_sizes;
  while (toker.valid_position()) {
    block_sizes.push_back(toker.scan_uint());
    toker.consume_char('\n');
  }

  const NodeID n = std::accumulate(block_sizes.begin(), block_sizes.end(), 0);

  StaticArray<BlockID> partition(n);
  NodeID cur = 0;
  BlockID block = 0;

  for (const NodeID block_size : block_sizes) {
    std::fill(partition.begin() + cur, partition.begin() + cur + block_size, block);
    cur += block_size;
    ++block;
  }

  return partition;
}

} // namespace partition

namespace remapping {

void write(const std::string &filename, std::span<const NodeID> remapping) {
  std::ofstream out(filename);
  for (const NodeID new_id : remapping) {
    out << new_id << "\n";
  }
}

} // namespace remapping

} // namespace kaminpar::shm::io
