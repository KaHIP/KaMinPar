#pragma once

#include "dynamic_graph_builder.h"
#include "graph_converter.h"
#include "mmio.h"

#include <cstdlib>
#include <fstream>

namespace kaminpar::tool::converter {
class SNAPReader : public GraphReader {
public:
  SimpleGraph read(const std::string &filename) override {
    std::ifstream in(filename);
    if (!in) { FATAL_PERROR << "Cannot open file " << filename; }
    std::string line;

    // read edges
    DynamicGraphBuilder builder;
    while (std::getline(in, line)) {
      if (line[0] == '#') { continue; } // skip comments

      const std::size_t sep{line.find_first_of('\t')};
      const auto from{static_cast<NodeID>(std::stol(line.substr(0, sep)))};
      const auto to{static_cast<NodeID>(std::stol(line.substr(sep + 1)))};
      builder.add_edge<true>(from, to, 1);
    }

    return builder.build();
  }

  [[nodiscard]] std::string description() const override { return "SNAP edge list format"; }
};
} // namespace kaminpar::tool::converter