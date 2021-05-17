#include "arguments.h"
#include "converter/edgelist_builder.h"
#include "utility/random.h"
#include "simple_graph.h"
#include "io.h"

using namespace kaminpar;
using namespace kaminpar::tool::converter;

int main(int argc, char *argv[]) {
  std::string filename{""};
  NodeID k_n{1024};
  double k_m_prob{0.8};
  NodeWeight k_min_weight{1};
  NodeWeight k_max_weight{100};
  NodeID k_n_deg_min{1};
  NodeID k_n_deg_max{5};
  NodeWeight min_low_deg_weight{1};
  NodeWeight max_low_deg_weight{10};

  Arguments args;
  args.positional()
      .argument("filename", "Output filename", &filename);
  args.group("Optional arguments")
      .argument("k-n", "Number of nodes in the kernel clique", &k_n)
      .argument("k-m-prob", "Probability for an edge in the kernel clique", &k_m_prob)
      .argument("k-min-weight", "Minimum weight of a kernel clique node", &k_min_weight)
      .argument("k-max-weight", "Maximum weight of a kernel clique node", &k_max_weight)
      .argument("k-n-deg-min", "Minimum number of 'low degree nodes' incident to each kernel clique node", &k_n_deg_min)
      .argument("k-n-deg-max", "Maximum number of 'low degree nodes' incident to each kernel clique node", &k_n_deg_max)
      .argument("low-deg-min-weight", "Minimum weight of a 'low degree node'", &min_low_deg_weight)
      .argument("low-deg-max-weight", "Maximum weight of a 'low degree node'", &max_low_deg_weight);
  args.parse(argc, argv);

  Randomize &rand{Randomize::instance()};
  EdgeListBuilder builder{k_n + k_n * k_n_deg_max};

  // generate clique
  for (NodeID u = 0; u < k_n; ++u) {
    for (NodeID v = u + 1; v < k_n; ++v) {
      if (rand.random_index(0, 100) < 100.0 * k_m_prob) {
        builder.add_edge(u, v, 1);
      }
    }

    const NodeID neighbors{static_cast<NodeID>(rand.random_index(k_n_deg_min, k_n_deg_max + 1))};
    for (NodeID v = 0; v < neighbors; ++v) {
      builder.add_edge(u, k_n + k_n_deg_max * u + v, 1);
    }
  }

  auto graph{builder.build()};
  graph.node_weights.resize(graph.n(), 1);
  graph.edge_weights.resize(graph.m(), 1);

  for (NodeID u = 0; u < k_n; ++u) {
    graph.node_weights[u] = rand.random_index(k_min_weight, k_max_weight + 1);
  }
  for (NodeID u = k_n; u < k_n + k_n * k_n_deg_max; ++u) {
    graph.node_weights[u] = rand.random_index(min_low_deg_weight, max_low_deg_weight + 1);
  }

  auto real_graph{simple_graph_to_graph(std::move(graph))};
  io::metis::write(filename, real_graph);
}