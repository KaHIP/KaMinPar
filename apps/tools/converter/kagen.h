#pragma once

#include <cstdlib>
#include <fstream>

#include "dynamic_graph_builder.h"
#include "graph_converter.h"
#include "mmio.h"

namespace kaminpar::tool::converter {
class KaGenReader : public GraphReader {
public:
  SimpleGraph read(const std::string &filename) override {
    std::ifstream in(filename);
    if (!in) { FATAL_PERROR << "Cannot open file " << filename; }
    std::string line;

    // read header
    std::getline(in, line);
    std::size_t n, m;
    {
      std::stringstream ss(line);
      char p;
      ss >> p >> n >> m;
      ASSERT(p == 'p') << "Expected file to start with 'p', but it starts with " << p;
    }

    // read edges
    DynamicGraphBuilder builder(n);
    while (std::getline(in, line)) {
      ASSERT(line[0] == 'e') << "Expected each line to start with 'e', but one starts with " << line[0];
      const std::string edge = line.substr(2);
      const std::size_t del = edge.find_first_of(' ');
      const auto from = static_cast<NodeID>(std::stol(edge.substr(0, del))) - 1;
      const auto to = static_cast<NodeID>(std::stol(edge.substr(del + 1))) - 1;
      if (from > to) {
        builder.add_edge<true>(from, to, 1);
      }
    }

    return builder.build();
  }

  [[nodiscard]] std::string description() const override { return "KaGen default edge list format"; }
};
} // namespace kaminpar::tool::converter