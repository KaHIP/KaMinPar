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

#include "apps/io/file_tokener.h"
#include "apps/io/metis_parser.h"
#include "apps/io/parhip_parser.h"
#include "apps/io/shm_compressed_graph_binary.h"

namespace kaminpar::shm::io {

std::unordered_map<std::string, GraphFileFormat> get_graph_file_formats() {
  return {
      {"metis", GraphFileFormat::METIS},
      {"parhip", GraphFileFormat::PARHIP},
  };
}

CSRGraph
csr_read(const std::string &filename, const GraphFileFormat file_format, const bool sorted) {
  switch (file_format) {
  case GraphFileFormat::METIS:
    return metis::csr_read(filename, sorted);
  case GraphFileFormat::PARHIP:
    return parhip::csr_read(filename, sorted);
  default:
    __builtin_unreachable();
  }
}

Graph read(
    const std::string &filename,
    const GraphFileFormat file_format,
    const bool compress,
    const bool may_dismiss,
    const NodeOrdering ordering
) {
  const bool sorted = ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS;

  if (compressed_binary::is_compressed(filename)) {
    if (!compress) {
      LOG_ERROR << "The input graph is stored in a compressed format but graph compression is"
                   "disabled!";
      std::exit(1);
    }

    return Graph(std::make_unique<CompressedGraph>(compressed_binary::read(filename)));
  }

  if (compress) {
    std::optional<CompressedGraph> compressed_graph = [&] {
      switch (file_format) {
      case GraphFileFormat::METIS: {
        return metis::compress_read(filename, sorted, may_dismiss);
      }
      case GraphFileFormat::PARHIP: {
        return std::optional(parhip::compressed_read_parallel(filename, ordering));
      }
      default:
        __builtin_unreachable();
      }
    }();

    if (compressed_graph) {
      return Graph(std::make_unique<CompressedGraph>(std::move(*compressed_graph)));
    }
  }

  return Graph(std::make_unique<CSRGraph>(csr_read(filename, file_format, sorted)));
}

namespace partition {

void write(const std::string &filename, const std::vector<BlockID> &partition) {
  std::ofstream out(filename);
  for (const BlockID block : partition) {
    out << block << "\n";
  }
}

std::vector<BlockID> read(const std::string &filename) {
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
