/*******************************************************************************
 * IO utilities for the shared-memory partitioner.
 *
 * @file:   shm_io.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "apps/io/shm_io.h"

#include <fstream>

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

    CompressedGraph compressed_graph = compressed_binary::read(filename);
    return Graph(std::make_unique<CompressedGraph>(std::move(compressed_graph)));
  }

  if (compress) {
    CompressedGraph compressed_graph = compressed_read(filename, file_format, ordering);
    return Graph(std::make_unique<CompressedGraph>(std::move(compressed_graph)));
  } else {
    CSRGraph csr_graph = csr_read(filename, file_format, ordering);
    return Graph(std::make_unique<CSRGraph>(std::move(csr_graph)));
  }
}

namespace partition {

void write(const std::string &filename, const std::vector<BlockID> &partition) {
  std::ofstream out(filename);
  for (const BlockID block : partition) {
    out << block << "\n";
  }
}

std::vector<BlockID> read(const std::string &filename) {
  using namespace kaminpar::io;
  MappedFileToker toker(filename);

  std::vector<BlockID> partition;
  while (toker.valid_position()) {
    partition.push_back(toker.scan_uint());
    toker.consume_char('\n');
  }

  return partition;
}

} // namespace partition

} // namespace kaminpar::shm::io
