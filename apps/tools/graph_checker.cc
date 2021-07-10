#include "definitions.h"
#include "utility/console_io.h"

#include <fstream>
#include <ranges>

// https://stackoverflow.com/questions/446296/where-can-i-get-a-useful-c-binary-search-algorithm
template<class Iter, class T>
Iter binary_find(Iter begin, Iter end, T val) {
  // Finds the lower bound in at most log(last - first) + 1 comparisons
  Iter i = std::lower_bound(begin, end, val);

  if (i != end && (*i <= val)) return i; // found
  else
    return end; // not found
}

using namespace kaminpar;

using idx = long long;

struct Edge {
  idx u;
  idx v;
  idx weight;

  bool operator<(const idx other_v) const { return v < other_v; }

  bool operator<=(const idx other_v) const { return v <= other_v; }
};

bool is_comment_line(const std::string &line) {
  const auto first_letter = line.find_first_not_of(" ");
  return first_letter != std::string::npos && line[first_letter] == '%';
}

int main(int argc, char *argv[]) {
  if (argc != 2) { FATAL_ERROR << "Usage: " << argv[0] << " <graph>"; }
  const std::string graph_filename{argv[1]};

  std::ifstream in(graph_filename);
  std::string line;
  while (std::getline(in, line) && is_comment_line(line)) {}

  idx n{0};
  idx m{0};
  idx format{0};
  {
    std::stringstream header(line);
    header >> n >> m >> format;
  }

  if (n < 0) { FATAL_ERROR << "Number of nodes cannot be negative."; }
  if (m < 0) { FATAL_ERROR << "Number of edges cannot be negative."; }
  if (m > n * (n - 1) / 2) {
    FATAL_ERROR << "There are too many edges: a graph with " << n << " nodes can have at most " << (n * (n - 1) / 2)
                << " undirected edges, but there are " << m << " undirected edges";
  }
  if (n > std::numeric_limits<NodeID>::max()) {
    FATAL_ERROR << "Number of nodes is too large (compiled with " << std::numeric_limits<NodeID>::digits
                << " bit get_datatype for node ids)";
  }
  if (m * 2 > std::numeric_limits<EdgeID>::max()) {
    FATAL_ERROR << "Number of edges is too large (compiled with " << std::numeric_limits<EdgeID>::digits
                << " bit get_datatype for edge ids); note that the graph has " << 2 * m << " directed edges";
  }
  if (format != 0 && format != 1 && format != 10 && format != 11) {
    FATAL_ERROR << "Unsupported format: supported formats are 0, 1, 10, 11, but given format is " << format;
  }

  const bool has_node_weights = (format % 100) / 10;
  const bool has_edge_weights = format % 10;
  m *= 2;

  cio::ProgressBar io_progress(m, "Reading graph");

  std::vector<Edge> edges;
  edges.reserve(m);
  idx total_node_weight{0};
  idx total_edge_weight{0};

  idx u = 0;
  while (std::getline(in, line)) {
    if (is_comment_line(line)) { continue; }
    std::stringstream node(line);

    if (has_node_weights) {
      idx weight;
      node >> weight;
      total_node_weight += weight;

      if (weight < 0) {
        FATAL_ERROR << "Node weight for node " << u + 1 << " must be at least 0";
      } else if (weight > std::numeric_limits<NodeWeight>::max()) {
        FATAL_ERROR << "Node weight for node " << u + 1 << " is too large: must be at most "
                    << std::numeric_limits<NodeWeight>::max();
      }
    }

    idx v;
    while (node >> v) {
      --v;

      if (v < 0) {
        FATAL_ERROR << "Invalid node id in the neighborhood of node " << u + 1 << ": " << v + 1
                    << " must be at least 1";
      } else if (v >= n) {
        FATAL_ERROR << "Invalid node id in the neighborhood of node " << u + 1 << ": " << v + 1
                    << " is higher than the number of nodes";
      } else if (v == u) {
        FATAL_ERROR << "Graph contains a loop on node " << u + 1;
      }

      idx weight{1};
      if (has_edge_weights) {
        node >> weight;

        if (weight <= 0) {
          FATAL_ERROR << "Edge weight for edge " << u + 1 << " --> " << v + 1 << " must be at least 1";
        } else if (weight > std::numeric_limits<EdgeWeight>::max()) {
          FATAL_ERROR << "Edge weight for edge " << u + 1 << " --> " << v + 1 << " too large: must be at most "
                      << std::numeric_limits<EdgeWeight>::max();
        }
      }

      edges.push_back({u, v, weight});
      io_progress.step();
    }
    ++u;
  }
  io_progress.stop();

  if (u < n) {
    FATAL_ERROR << "Number of nodes mismatches: header specifies " << n << " nodes, but there are only " << u
                << " nodes in the file";
  }
  if (static_cast<idx>(edges.size()) != m) {
    FATAL_ERROR << "Number of edges mismatches: header specifies " << m << " undirected edges, but there are "
                << edges.size() << " undirected edges in the file";
  }

  if (total_node_weight > std::numeric_limits<NodeWeight>::max()) {
    FATAL_ERROR << "Sum of node weights exceeds " << std::numeric_limits<NodeWeight>::digits << " bits";
  }
  if (total_edge_weight > std::numeric_limits<EdgeWeight>::max()) {
    FATAL_ERROR << "Sum of edge weights exceeds " << std::numeric_limits<EdgeWeight>::digits << " bits";
  }

  cio::ProgressBar edge_checking_progress(5, "Checking structure");

  edge_checking_progress.step("sorting edges");
  std::ranges::sort(edges, [](const auto &a, const auto &b) { return a.u < b.u || (a.u == b.u && a.v < b.v); });

  edge_checking_progress.step("checking for duplicate edges");
  for (std::size_t i = 1; i < edges.size(); ++i) {
    const Edge &prev = edges[i - 1];
    const Edge &cur = edges[i];
    if (prev.u == cur.u && prev.v == cur.v) {
      FATAL_ERROR << "Duplicate edge: " << cur.u << " --> " << cur.v << " with weights " << cur.weight << ", "
                  << prev.weight;
    }
  }

  edge_checking_progress.step("computing index structure");
  std::vector<NodeID> nodes(n + 1);
  for (idx i = 0, j = 0; i < n; ++i) {
    while (j < m && edges[j].u == i) { ++j; }
    nodes[i + 1] = j;
  }

  edge_checking_progress.step("checking for reverse edges");
  for (const auto &[u, v, weight] : edges) {
    if (u > v) { continue; }
    const auto end = edges.begin() + nodes[v + 1];
    const auto rev_edge = binary_find(edges.begin() + nodes[v], end, u);
    if (rev_edge == end) {
      FATAL_ERROR << "There is a directed edge " << u + 1 << " --> " << v + 1
                  << ", but the reverse edge does not exist";
    }

    if (weight != rev_edge->weight) {
      FATAL_ERROR << "The directed edge " << u + 1 << " --> " << v + 1 << " has weight " << weight
                  << ", but the reverse edge has weight " << rev_edge->weight;
    }
  }
  edge_checking_progress.stop();

  return 0;
}