#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <kaminpar-common/random.h>

#include <kaminpar-io/io.h>

#include <kaminpar-shm/datastructures/graph.h>
#include <kaminpar-shm/kaminpar.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace kaminpar::shm {

PYBIND11_MODULE(kaminpar_python, m) {
  using namespace pybind11::literals;

  m.doc() = "Python Bindings for KaMinPar";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  m.def("seed", &Random::get_seed, "The seed for the random number generator");
  m.def("reseed", &Random::reseed, "Reseed the random number generator");

  pybind11::class_<Context>(m, "Context");
  m.def("context_names", &get_preset_names, "All available context names");
  m.def(
      "context_by_name",
      [](const std::string &context_name) {
        try {
          return create_context_by_preset_name(context_name);
        } catch (const std::exception &e) {
          throw std::invalid_argument("Invalid context name.");
        }
      },
      "Create a context by name"
  );

  m.def("default_context", &create_default_context, "Create the default context");
  m.def("fast_context", &create_fast_context, "Create the faster context");
  m.def("strong_context", &create_strong_context, "Create the higher-quality context");

  m.def(
      "terapart_context",
      &create_terapart_context,
      "Create the default context for memory-efficient partitioning"
  );
  m.def(
      "terapart_strong_context",
      &create_terapart_strong_context,
      "Create the higher-qulity context for memory-efficient partitioning"
  );
  m.def(
      "terapart_largek_context",
      &create_terapart_largek_context,
      "Create the context for memory-efficient large-k partitioning"
  );

  m.def(
      "largek_context",
      &create_largek_context,
      "Create the default context for large-k partitioning"
  );
  m.def(
      "largek_fast_context",
      &create_largek_fast_context,
      "Create the fast context for large-k partitioning"
  );
  m.def(
      "largek_strong_context",
      &create_largek_strong_context,
      "Create the higher-quality context for large-k partitioning"
  );

  pybind11::class_<Graph>(m, "Graph")
      .def("n", &Graph::n, "Number of nodes")
      .def("m", &Graph::m, "Number of edges")
      .def(
          "nodes",
          [](const Graph &self) {
            auto range = self.nodes();
            return pybind11::make_iterator(range.begin(), range.end());
          },
          ""
      )
      .def(
          "edges",
          [](const Graph &self) {
            auto range = self.edges();
            return pybind11::make_iterator(range.begin(), range.end());
          },
          ""
      )
      .def("is_node_weighted", &Graph::is_node_weighted, "Whether the graph has node weights")
      .def("is_edge_weighted", &Graph::is_edge_weighted, "Whether the graph has edge weights")
      .def("node_weight", &Graph::node_weight, "The weight of a node")
      .def("degree", &Graph::degree, "The degree of a node")
      .def(
          "neighbors",
          [](const Graph &self, NodeID u) {
            std::vector<std::pair<NodeID, EdgeWeight>> neighbors;
            neighbors.reserve(self.degree(u));

            self.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
              neighbors.emplace_back(v, w);
            });

            return neighbors;
          },
          "The neighbors of a node"
      );

  pybind11::enum_<io::GraphFileFormat>(m, "GraphFileFormat")
      .value("METIS", io::GraphFileFormat::METIS)
      .value("PARHIP", io::GraphFileFormat::PARHIP);
  m.def(
      "load_graph",
      [](const std::string &filename, io::GraphFileFormat format) -> Graph {
        if (auto graph = io::read(filename, format, NodeOrdering::NATURAL, false)) {
          return std::move(*graph);
        }

        throw std::invalid_argument("Failed to load graph");
      },
      "Loads a graph from a file"
  );

  pybind11::class_<KaMinPar>(m, "KaMinPar")
      .def(pybind11::init<int, Context>(), "num_threads"_a, "ctx"_a)
      .def(
          "compute_partition",
          [](KaMinPar &self, Graph &graph, const BlockID k, const double epsilon) {
            // Disable node ordering as the graph is modified in the process.
            self.context().node_ordering = NodeOrdering::NATURAL;

            std::vector<BlockID> partition(graph.n());
            self.set_graph(std::move(graph));
            self.compute_partition(k, partition);

            graph = self.take_graph();
            return partition;
          },
          "Compute a partition",
          "graph"_a,
          "k"_a,
          "eps"_a = 0.03
      );
}

} // namespace kaminpar::shm
