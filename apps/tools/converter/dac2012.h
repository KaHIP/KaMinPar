#pragma once

#include "dynamic_graph_builder.h"
#include "graph_converter.h"
#include "mmio.h"

#include <cstdlib>
#include <fstream>

namespace kaminpar::tool::converter {
class Dac2012Reader : public GraphReader {
  static constexpr bool kDebug = false;

public:
  SimpleGraph read(const std::string &filename) override {
    std::ifstream in(filename);
    if (!in) { FATAL_PERROR << "Cannot open file " << filename; }
    std::string line;

    std::unordered_map<std::string, NodeID> node_map;

    bool parsing_net{false};
    NodeID num_nets{0};
    NodeID from_node{0};

    DynamicGraphBuilder builder{};

    while (std::getline(in, line)) {
      if (num_nets == 0 && line.rfind("NumNets", 0) == 0) {
        const auto sep = line.find(":");
        auto str_num_nets = line.substr(sep + 1);
        utility::str::trim(str_num_nets);
        num_nets = std::stoi(str_num_nets);
        DBG << "Number of nets: " << num_nets;
      } else {
        if (line.rfind("NetDegree", 0) == 0) { // start of new net
          DBG << "New net " << line;
          if (parsing_net) { ++from_node; } // dont increment the first time
          parsing_net = true;
          ASSERT(from_node < num_nets);
        } else if (parsing_net) {
          utility::str::ltrim(line);
          const auto space_sep = line.find(" ");
          const auto to_node_name = line.substr(0, space_sep);

          if (!node_map.contains(to_node_name)) {
            node_map[to_node_name] = num_nets + node_map.size();
          }
          const auto to_node{node_map[to_node_name]};

          builder.add_edge<true>(from_node, to_node, 1);

          DBG << "Add edge from " << from_node << " to " << to_node << " (" << to_node_name << ")";
        }
      }
    }

    return builder.build();
  }

  [[nodiscard]] std::string description() const override { return "Reader for net lists from DAC2012 benchmark set. Builds a bipartite graph: net to pins."; }
};
} // namespace kaminpar::tool::converter