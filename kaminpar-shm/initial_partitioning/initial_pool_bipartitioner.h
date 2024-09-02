/*******************************************************************************
 * Initial partitioner that uses a portfolio of initial partitioning algorithms.
 * Each graphutils is repeated multiple times. Algorithms that are unlikely to
 * beat the best partition found so far are executed less often than promising
 * candidates.
 *
 * @file:   initial_pool_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <memory>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/initial_partitioning/initial_flat_bipartitioner.h"
#include "kaminpar-shm/initial_partitioning/initial_refiner.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::shm {
class InitialPoolBipartitioner {
public:
  struct RunningVariance {
    [[nodiscard]] std::pair<double, double> get() const;
    void reset();
    void update(double value);

    std::size_t _count = 0;
    double _mean = 0.0;
    double _M2 = 0.0;
  };

  struct BipartitionerStatistics {
    void reset();

    std::vector<EdgeWeight> cuts;
    double cut_mean;
    double cut_variance;
    std::size_t num_feasible_partitions;
    std::size_t num_infeasible_partitions;
  };

  struct Statistics {
    std::vector<BipartitionerStatistics> per_bipartitioner;
    EdgeWeight best_cut;
    std::size_t best_bipartitioner;
    bool best_feasible;
    double best_imbalance;
    std::size_t num_balanced_partitions;
    std::size_t num_imbalanced_partitions;
  };

  InitialPoolBipartitioner(const InitialPoolPartitionerContext &pool_ctx);

  void set_num_repetitions(int num_repetitions);

  void init(const CSRGraph &graph, const PartitionContext &p_ctx);

  template <typename BipartitionerType> void register_bipartitioner(const std::string_view name) {
    KASSERT(
        std::find(_bipartitioner_names.begin(), _bipartitioner_names.end(), name) ==
        _bipartitioner_names.end()
    );

    _bipartitioners.push_back(std::make_unique<BipartitionerType>(_pool_ctx));
    _bipartitioner_names.push_back(name);
    _running_statistics.emplace_back();
    _statistics.per_bipartitioner.emplace_back();
  }

  void reset();

  PartitionedCSRGraph bipartition();

private:
  void run_bipartitioner(std::size_t i);

  [[nodiscard]] bool likely_to_improve(std::size_t i) const;

  void finalize_statistics();

  void print_statistics();

  const CSRGraph *_graph;
  const PartitionContext *_p_ctx;

  const InitialPoolPartitionerContext &_pool_ctx;
  int _num_repetitions;

  StaticArray<BlockID> _best_partition{0, static_array::small, static_array::seq};
  StaticArray<BlockID> _current_partition{0, static_array::small, static_array::seq};

  StaticArray<BlockWeight> _best_block_weights{2, 0, static_array::small, static_array::seq};
  StaticArray<BlockWeight> _current_block_weights{2, 0, static_array::small, static_array::seq};

  EdgeWeight _best_cut = std::numeric_limits<EdgeWeight>::max();
  bool _best_feasible = false;
  double _best_imbalance = 0.0;
  std::size_t _best_bipartitioner = 0;

  std::vector<std::string_view> _bipartitioner_names{};
  std::vector<std::unique_ptr<InitialFlatBipartitioner>> _bipartitioners{};

  std::unique_ptr<InitialRefiner> _refiner;

  std::vector<RunningVariance> _running_statistics{};
  Statistics _statistics{};
};
} // namespace kaminpar::shm
