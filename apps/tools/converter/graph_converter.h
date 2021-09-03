#pragma once

#include "simple_graph.h"

#include <concepts>
#include <memory>
#include <string>
#include <unordered_map>

namespace kaminpar::tool::converter {
class GraphReader {
public:
  virtual ~GraphReader() = default;
  virtual SimpleGraph read(const std::string &filename) = 0;
  [[nodiscard]] virtual std::string description() const = 0;
  [[nodiscard]] virtual std::string default_extension() const { return ""; }
};

class GraphWriter {
public:
  virtual ~GraphWriter() = default;
  virtual void write(const std::string &filename, SimpleGraph graph, const std::string &comment) = 0;
  [[nodiscard]] virtual std::string description() const = 0;
  [[nodiscard]] virtual std::string default_extension() const { return ""; }
};

class GraphProcessor {
public:
  virtual ~GraphProcessor() = default;
  virtual void process(SimpleGraph &graph) = 0;
  [[nodiscard]] virtual std::string description() const = 0;
};

class GraphConverter {
public:
  template<typename Reader, typename... Args>
  requires std::derived_from<Reader, GraphReader>
  void register_reader(const std::string &name, Args &&...args) {
    _readers[name] = std::make_unique<Reader>(std::forward<Args>(args)...);
  }

  template<typename Writer, typename... Args>
  requires std::derived_from<Writer, GraphWriter>
  void register_writer(const std::string &name, Args &&...args) {
    _writers[name] = std::make_unique<Writer>(std::forward<Args>(args)...);
  }

  template<typename Processor, typename... Args>
  requires std::derived_from<Processor, GraphProcessor>
  void register_processor(const std::string &name, Args &&...args) {
    _processors[name] = std::make_unique<Processor>(std::forward<Args>(args)...);
  }

  [[nodiscard]] bool reader_exists(const std::string &name) const {
    if (_readers.contains(name)) {
      return true;
    }

    for (const auto &[ignored, reader] : _readers) {
      if (reader->default_extension() == name) { return true; }
    }

    return false;
  }
  [[nodiscard]] bool writer_exists(const std::string &name) const { return _writers.contains(name); }
  [[nodiscard]] auto &get_readers() const { return _readers; }
  [[nodiscard]] auto &get_writers() const { return _writers; }
  [[nodiscard]] auto &get_processors() const { return _processors; }

  [[nodiscard]] std::unique_ptr<GraphReader> &get_reader(const std::string &name) {
    if (!_readers.contains(name)) {
      for (auto &[ignored, reader] : _readers) {
        if (reader->default_extension() == name) { return reader; }
      }
      FATAL_ERROR << "unknown reader " << name;
      __builtin_unreachable();
    } else {
      return _readers[name];
    }
  }
  [[nodiscard]] auto &get_writer(const std::string &name) { return _writers[name]; }

  void set_comment(const std::string &comment) { _comment = comment; }

  void convert(const std::string &importer,     //
               const std::string &in_filename,  //
               const std::string &exporter,     //
               const std::string &out_filename, //
               const std::string &processors) { //
    std::vector<GraphProcessor *> processor_ptrs;
    {
      std::stringstream processor_names(processors);
      std::string processor_name;
      while (std::getline(processor_names, processor_name, ',')) {
        if (!_processors.contains(processor_name)) { FATAL_ERROR << "Unknown graph processor: " << processor_name; }
        processor_ptrs.push_back(_processors[processor_name].get());
      }
    }

    LOG << "Reading graph from " << in_filename << " ...";
    SimpleGraph graph = get_reader(importer)->read(in_filename);
    if (!processor_ptrs.empty()) { LOG << "Processing ..."; }
    for (GraphProcessor *processor : processor_ptrs) { processor->process(graph); }
    LOG << "Writing graph to " << out_filename << " ...";
    if (!_comment.empty()) {
      _writers[exporter]->write(out_filename, graph, _comment);
    } else {
      _writers[exporter]->write(out_filename, graph, "");
    }
  }

private:
  std::unordered_map<std::string, std::unique_ptr<GraphReader>> _readers{};
  std::unordered_map<std::string, std::unique_ptr<GraphWriter>> _writers{};
  std::unordered_map<std::string, std::unique_ptr<GraphProcessor>> _processors{};
  std::string _comment{};
};
} // namespace kaminpar::tool::converter