#include "io.h"

#include "utility/timer.h"

#include <cctype>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kaminpar::io {
static constexpr auto kDebug = false;

namespace {
struct MappedFile {
  const int fd;
  std::size_t position;
  const std::size_t length;
  char *contents;

  [[nodiscard]] inline bool valid_position() const { return position < length; }
  [[nodiscard]] inline char current() const { return contents[position]; }
  inline void advance() { ++position; }
};

int open_file(const std::string &filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) FATAL_PERROR << "Error while opening " << filename;
  return fd;
}

std::size_t file_size(const int fd) {
  struct stat file_info {};
  if (fstat(fd, &file_info) == -1) {
    close(fd);
    FATAL_PERROR << "Error while determining file size";
  }
  return static_cast<std::size_t>(file_info.st_size);
}

MappedFile mmap_file_from_disk(const std::string &filename) {
  const int fd = open_file(filename);
  const std::size_t length = file_size(fd);

  char *contents = static_cast<char *>(mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0));
  if (contents == MAP_FAILED) {
    close(fd);
    FATAL_PERROR << "Error while mapping file to memory";
  }

  return {
      .fd = fd,
      .position = 0,
      .length = length,
      .contents = contents,
  };
}

void munmap_file_from_disk(const MappedFile &mapped_file) {
  if (munmap(mapped_file.contents, mapped_file.length) == -1) {
    close(mapped_file.fd);
    FATAL_PERROR << "Error while unmapping file from memory";
  }
  close(mapped_file.fd);
}

inline void skip_spaces(MappedFile &mapped_file) {
  while (mapped_file.valid_position() && std::isspace(mapped_file.current())) { mapped_file.advance(); }
}

inline void skip_comment(MappedFile &mapped_file) {
  while (mapped_file.valid_position() && mapped_file.current() != '\n') { mapped_file.advance(); }
  if (mapped_file.valid_position()) {
    ASSERT(mapped_file.current() == '\n');
    mapped_file.advance();
  }
}

inline void skip_nl(MappedFile &mapped_file) {
  ASSERT(mapped_file.valid_position() && mapped_file.current() == '\n');
  mapped_file.advance();
}

inline std::uint64_t scan_uint(MappedFile &mapped_file) {
  std::uint64_t number = 0;
  while (mapped_file.valid_position() && std::isdigit(mapped_file.current())) {
    const int digit = mapped_file.current() - '0';
    number = number * 10 + digit;
    mapped_file.advance();
  }
  skip_spaces(mapped_file);
  return number;
}
} // namespace

//
// Internal Metis functions
//

namespace metis {
struct GraphHeader {
  uint64_t number_of_nodes;
  uint64_t number_of_edges;
  bool has_node_weights;
  bool has_edge_weights;
};

GraphHeader read_graph_header(MappedFile &mapped_file) {
  skip_spaces(mapped_file);
  while (mapped_file.current() == '%') {
    skip_comment(mapped_file);
    skip_spaces(mapped_file);
  }

  const std::uint64_t number_of_nodes = scan_uint(mapped_file);
  const std::uint64_t number_of_edges = scan_uint(mapped_file);
  const std::uint64_t format = (mapped_file.current() != '\n') ? scan_uint(mapped_file) : 0;
  skip_nl(mapped_file);

  [[maybe_unused]] const bool has_node_sizes = format / 100; // == 1xx
  const bool has_node_weights = (format % 100) / 10;         // == x1x
  const bool has_edge_weights = format % 10;                 // == xx1

  ASSERT(!has_node_sizes); // unsupported
  return {
      .number_of_nodes = number_of_nodes,
      .number_of_edges = number_of_edges,
      .has_node_weights = has_node_weights,
      .has_edge_weights = has_edge_weights,
  };
}

GraphHeader parse_header(const std::string &filename) {
  MappedFile mapped_file = mmap_file_from_disk(filename);
  const metis::GraphHeader header = metis::read_graph_header(mapped_file);
  munmap_file_from_disk(mapped_file);
  return header;
}

void write_file(std::ofstream &out, const StaticArray<EdgeID> &nodes, const StaticArray<NodeID> &edges,
                const StaticArray<NodeWeight> &node_weights, const StaticArray<EdgeWeight> &edge_weights,
                const std::string &comment) {
  const bool write_node_weights = !node_weights.empty();
  const bool write_edge_weights = !edge_weights.empty();

  if (!comment.empty()) { out << "% " << comment << "\n"; }

  // header
  out << nodes.size() - 1 << " " << edges.size() / 2;
  if (write_node_weights || write_edge_weights) {
    out << " " << static_cast<int>(write_node_weights) << static_cast<int>(write_edge_weights);
  }
  out << "\n";

  // content
  for (NodeID u = 0; u < nodes.size() - 1; ++u) {
    if (write_node_weights) { out << node_weights[u] << " "; }
    for (EdgeID e = nodes[u]; e < nodes[u + 1]; ++e) {
      out << edges[e] + 1 << " ";
      if (write_edge_weights) { out << edge_weights[e] << " "; }
    }
    out << "\n";
  }
}

void write_file(const std::string &filename, const StaticArray<EdgeID> &nodes, const StaticArray<NodeID> &edges,
                const StaticArray<NodeWeight> &node_weights, const StaticArray<EdgeWeight> &edge_weights,
                const std::string &comment) {
  std::ofstream out(filename);
  if (!out) { FATAL_PERROR << "Error while opening " << filename; }
  write_file(out, nodes, edges, node_weights, edge_weights, comment);
}
} // namespace metis

//
// Public Metis functions
//

namespace metis {
void read_format(const std::string &filename, NodeID &n, EdgeID &m, bool &has_node_weights, bool &has_edge_weights) {
  MappedFile mapped_file = mmap_file_from_disk(filename);
  const metis::GraphHeader header = metis::read_graph_header(mapped_file);
  munmap_file_from_disk(mapped_file);

  n = header.number_of_nodes;
  m = header.number_of_edges;
  has_node_weights = header.has_node_weights;
  has_edge_weights = header.has_edge_weights;
}

GraphInfo read(const std::string &filename, StaticArray<EdgeID> &nodes, StaticArray<NodeID> &edges,
               StaticArray<NodeWeight> &node_weights, StaticArray<EdgeWeight> &edge_weights) {
  GraphInfo info{};

  MappedFile mapped_file = mmap_file_from_disk(filename);
  const metis::GraphHeader header = metis::read_graph_header(mapped_file);
  const bool read_node_weights = header.has_node_weights;
  const bool read_edge_weights = header.has_edge_weights;

  nodes.resize(header.number_of_nodes + 1);
  edges.resize(header.number_of_edges * 2);
  if (read_node_weights) { node_weights.resize(header.number_of_nodes); }
  if (read_edge_weights) { edge_weights.resize(header.number_of_edges * 2); }

  bool unit_node_weights = true;
  bool unit_edge_weights = true;

  EdgeID e = 0;
  for (NodeID u = 0; u < header.number_of_nodes; ++u) {
    const EdgeID e_before_u = e;
    nodes[u] = e;

    skip_spaces(mapped_file);
    while (mapped_file.current() == '%') {
      skip_comment(mapped_file);
      skip_spaces(mapped_file);
    }

    if (header.has_node_weights) {
      if (read_node_weights) {
        node_weights[u] = scan_uint(mapped_file);
        unit_node_weights = unit_node_weights && node_weights[u] == 1;
        info.total_node_weight += node_weights[u];
      } else {
        scan_uint(mapped_file);
      }
    }

    while (std::isdigit(mapped_file.current())) {
      edges[e] = scan_uint(mapped_file) - 1;
      if (header.has_edge_weights) {
        if (read_edge_weights) {
          edge_weights[e] = scan_uint(mapped_file);
          unit_edge_weights = unit_edge_weights && edge_weights[e] == 1;
          info.total_edge_weight += edge_weights[e];
        } else {
          scan_uint(mapped_file);
        }
      }
      ++e;
    }
    info.has_isolated_nodes |= (e_before_u == e);

    if (mapped_file.current() == '\n') { skip_nl(mapped_file); }
  }
  nodes.back() = e;

  munmap_file_from_disk(mapped_file);
  ASSERT(e == 2 * header.number_of_edges);

  // only keep weights if the graph is actually weighted
  if (unit_node_weights) {
    node_weights.free();
    info.total_node_weight = header.number_of_nodes;
  }

  if (unit_edge_weights) {
    edge_weights.free();
    info.total_edge_weight = 2 * header.number_of_edges;
  }

  return info;
}

Graph read(const std::string &filename, bool ignore_node_weights, bool ignore_edge_weights) {
  StaticArray<EdgeID> nodes;
  StaticArray<NodeID> edges;
  StaticArray<NodeWeight> node_weights;
  StaticArray<EdgeWeight> edge_weights;
  metis::read(filename, nodes, edges, node_weights, edge_weights);

  if (ignore_node_weights) { node_weights.free(); }
  if (ignore_edge_weights) { edge_weights.free(); }

  return Graph(std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights));
}

void write(const std::string &filename, const Graph &graph, const std::string &comment) {
  metis::write_file(filename, graph.raw_nodes(), graph.raw_edges(), graph.raw_node_weights(), graph.raw_edge_weights(),
                    comment);
}
} // namespace metis

//
// Partition
//

namespace partition {
void write(const std::string &filename, const StaticArray<BlockID> &partition) {
  std::ofstream out(filename);
  for (const BlockID block : partition) { out << block << "\n"; }
}

void write(const std::string &filename, const PartitionedGraph &p_graph) { write(filename, p_graph.partition()); }

void write(const std::string &filename, const StaticArray<BlockID> &partition, const NodePermutation &permutation) {
  std::ofstream out(filename);
  for (const NodeID u : permutation) { out << partition[u] << "\n"; }
}

void write(const std::string &filename, const PartitionedGraph &p_graph, const NodePermutation &permutation) {
  write(filename, p_graph.partition(), permutation);
}

std::vector<BlockID> read(const std::string &filename) {
  MappedFile mapped_file = mmap_file_from_disk(filename);
  std::vector<BlockID> partition;
  while (mapped_file.valid_position()) {
    partition.push_back(scan_uint(mapped_file));
    skip_nl(mapped_file);
  }
  munmap_file_from_disk(mapped_file);
  return partition;
}
} // namespace partition
} // namespace kaminpar::io
