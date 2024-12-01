/*******************************************************************************
 * Interface for refinement algorithms.
 *
 * @file:   refiner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <string_view>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

class Refiner {
public:
  enum class OutputLevel : std::uint8_t {
    QUIET = 0,
    INFO = 1,
    DEBUG = 2,
  };

  Refiner(const Refiner &) = delete;
  Refiner &operator=(const Refiner &) = delete;

  Refiner(Refiner &&) noexcept = default;
  Refiner &operator=(Refiner &&) noexcept = default;

  virtual ~Refiner() = default;

  [[nodiscard]] virtual std::string name() const {
    return "Unknown";
  }

  virtual void initialize(const PartitionedGraph &p_graph) = 0;

  virtual bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;

  void set_output_prefix(const std::string_view prefix) {
    _output_prefix = prefix;
  }

  void set_output_level(const OutputLevel level) {
    _output_level = level;
  }

protected:
  Refiner() = default;

  OutputLevel _output_level = OutputLevel::QUIET;
  std::string_view _output_prefix = "";
};

class NoopRefiner : public Refiner {
public:
  void initialize(const PartitionedGraph &) final {}

  bool refine(PartitionedGraph &, const PartitionContext &) final {
    return false;
  }
};

} // namespace kaminpar::shm
