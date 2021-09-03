#pragma once

#include "definitions.h"
#include "graph_converter.h"
#include "io.h"

#include <fstream>
#include <memory>

namespace kaminpar::tool::converter {
class MetisReader : public GraphReader {
public:
  SimpleGraph read(const std::string &filename) override {
    return graph_to_simple_graph(io::metis::read(filename));
  }

  [[nodiscard]] std::string description() const override { return "METIS graph format"; }

  [[nodiscard]] std::string default_extension() const override { return "graph"; }
};

class MetisWriter : public GraphWriter {
public:
  void write(const std::string &filename, SimpleGraph graph, const std::string &comment) override {
    LOG << "Saving graph in METIS format ...";
    io::metis::write(filename, simple_graph_to_graph(graph), comment);
  }

  [[nodiscard]] std::string description() const override { return "METIS graph format"; }

  [[nodiscard]] std::string default_extension() const override { return "graph"; }
};
} // namespace kaminpar::tool::converter
