/*******************************************************************************
 * IO utilities for the shared-memory partitioner.
 *
 * @file:   shm_io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "apps/io/shm_io.h"

#include <fstream>
#include <numeric>

#include "kaminpar-common/logger.h"

#include "apps/io/file_toker.h"
#include "apps/io/shm_compressed_graph_binary.h"
#include "apps/io/shm_metis_parser.h"
#include "apps/io/shm_parhip_parser.h"

namespace kaminpar::shm::io {

std::unordered_map<std::string, GraphFileFormat> get_graph_file_formats() {
  return {
      {"metis", GraphFileFormat::METIS},
      {"parhip", GraphFileFormat::PARHIP},
  };
}

CSRGraph csr_read(
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
  case GraphFileFormat::PARHIP: {
    return parhip::csr_read(filename, ordering);
  }
  default:
    __builtin_unreachable();
  }
}

CompressedGraph compressed_read(
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
  case GraphFileFormat::PARHIP: {
    return parhip::compressed_read_parallel(filename, ordering);
  }
  default:
    __builtin_unreachable();
  }
}

Graph read(
    const std::string &filename,
    const GraphFileFormat file_format,
    const NodeOrdering ordering,
    const bool compress
) {
  if (compressed_binary::is_compressed(filename)) {
    if (!compress) {
      LOG_ERROR << "The input graph is stored in a compressed format but graph compression is "
                   "disabled!";
      std::exit(EXIT_FAILURE);
    }

    return {std::make_unique<CompressedGraph>(compressed_binary::read(filename))};
  }

  if (compress) {
    return {std::make_unique<CompressedGraph>(compressed_read(filename, file_format, ordering))};
  } else {
    return {std::make_unique<CSRGraph>(csr_read(filename, file_format, ordering))};
  }
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

} // namespace kaminpar::shm::io
