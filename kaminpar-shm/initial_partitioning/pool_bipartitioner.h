/*******************************************************************************
 * Initial partitioner that uses a portfolio of initial partitioning algorithms.
 * Each graphutils is repeated multiple times. Algorithms that are unlikely to
 * beat the best partition found so far are executed less often than promising
 * candidates.
 *
 * @file:   pool_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/initial_partitioning/bfs_bipartitioner.h"
#include "kaminpar-shm/initial_partitioning/bipartitioner.h"
#include "kaminpar-shm/initial_partitioning/greedy_graph_growing_bipartitioner.h"
#include "kaminpar-shm/initial_partitioning/initial_refiner.h"
#include "kaminpar-shm/initial_partitioning/random_bipartitioner.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::shm::ip {
class PoolBipartitioner {
  SET_DEBUG(false);

  friend class PoolBipartitionerFactory;

  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  struct RunningVariance {
    [[nodiscard]] std::pair<double, double> get() const {
      if (_count == 0) {
        return {std::numeric_limits<double>::max(), 0.0};
      } else if (_count < 2) {
        return {_mean, 0.0};
      } else {
        return {_mean, _M2 / _count};
      }
    }

    void reset() {
      _mean = 0.0;
      _count = 0;
      _M2 = 0.0;
    }

    void update(const double value) {
      ++_count;
      double delta = value - _mean;
      _mean += delta / _count;
      double delta2 = value - _mean;
      _M2 += delta * delta2;
    }

    std::size_t _count{0};
    double _mean{0.0};
    double _M2{0.0};
  };

public:
  struct BipartitionerStatistics {
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

  PoolBipartitioner(const InitialPartitioningContext &i_ctx)
      : _i_ctx(i_ctx),
        _min_num_repetitions(i_ctx.min_num_repetitions),
        _min_num_non_adaptive_repetitions(i_ctx.min_num_non_adaptive_repetitions),
        _max_num_repetitions(i_ctx.max_num_repetitions) {
    register_bipartitioner<GreedyGraphGrowingBipartitioner>("greedy_graph_growing");
    register_bipartitioner<AlternatingBfsBipartitioner>("bfs_alternating");
    register_bipartitioner<LighterBlockBfsBipartitioner>("bfs_lighter_block");
    register_bipartitioner<LongerQueueBfsBipartitioner>("bfs_longer_queue");
    register_bipartitioner<ShorterQueueBfsBipartitioner>("bfs_shorter_queue");
    register_bipartitioner<SequentialBfsBipartitioner>("bfs_sequential");
    register_bipartitioner<RandomBipartitioner>("random");
  }

  void init(const CSRGraph &graph, const PartitionContext &p_ctx, InitialRefiner *refiner) {
    _graph = &graph;
    _p_ctx = &p_ctx;
    _refiner = refiner;

    _refiner->init(*_graph);
    for (auto &bipartitioner : _bipartitioners) {
      bipartitioner->init(*_graph, *_p_ctx);
    }

    if (_current_partition.size() < _graph->n()) {
      _current_partition.resize(_graph->n(), static_array::small, static_array::seq);
    }
    if (_best_partition.size() < _graph->n()) {
      _best_partition.resize(_graph->n(), static_array::small, static_array::seq);
    }

    reset();
  }

  template <typename BipartitionerType> void register_bipartitioner(const std::string_view name) {
    KASSERT(
        std::find(_bipartitioner_names.begin(), _bipartitioner_names.end(), name) ==
        _bipartitioner_names.end()
    );

    _bipartitioners.push_back(std::make_unique<BipartitionerType>(_i_ctx));
    _bipartitioner_names.push_back(name);
    _running_statistics.emplace_back();
    _statistics.per_bipartitioner.emplace_back();
  }

  const Statistics &statistics() {
    return _statistics;
  }

  void reset() {
    const std::size_t n = _bipartitioners.size();

    _running_statistics.clear();
    _running_statistics.resize(n);

    _statistics.per_bipartitioner.clear();
    _statistics.per_bipartitioner.resize(n);

    _best_feasible = false;
    _best_cut = std::numeric_limits<EdgeWeight>::max();
    _best_imbalance = 0.0;
  }

  PartitionedCSRGraph bipartition() {
    KASSERT(_current_partition.size() >= _graph->n());
    KASSERT(_best_partition.size() >= _graph->n());

    // only perform more repetitions with bipartitioners that are somewhat
    // likely to find a better partition than the current one
    const auto repetitions =
        std::clamp(_num_repetitions, _min_num_repetitions, _max_num_repetitions);
    for (std::size_t rep = 0; rep < repetitions; ++rep) {
      for (std::size_t i = 0; i < _bipartitioners.size(); ++i) {
        if (rep < _min_num_non_adaptive_repetitions ||
            !_i_ctx.use_adaptive_bipartitioner_selection || likely_to_improve(i)) {
          run_bipartitioner(i);
        }
      }
    }

    finalize_statistics();
    if constexpr (kDebug) {
      print_statistics();
    }

    // To avoid copying the partition / block weights, we create non-owning views of
    // our cached data structures; this assumes a lot about the calling code and is
    // a bit dangerous, but yolo.
    //
    // (a) the lifespan of this object must be exceed the lifespan of the partitioned
    // graph
    // (b) this partitioner may not be used until the partitioned graph is destroyed
    //
    // @todo un-yolo-fy this
    StaticArray<BlockID> best_partition_view(_graph->n(), _best_partition.data());
    StaticArray<BlockWeight> best_block_weights_view(2, _best_block_weights.data());
    return {*_graph, 2, std::move(best_partition_view), std::move(best_block_weights_view)};
  }

  void set_num_repetitions(const std::size_t num_repetitions) {
    _num_repetitions = num_repetitions;
  }

private:
  bool likely_to_improve(const std::size_t i) const {
    const auto [mean, variance] = _running_statistics[i].get();
    const double rhs = (mean - static_cast<double>(_best_cut)) / 2;
    return variance > rhs * rhs;
  }

  void finalize_statistics() {
    for (std::size_t i = 0; i < _bipartitioners.size(); ++i) {
      const auto [mean, variance] = _running_statistics[i].get();
      _statistics.per_bipartitioner[i].cut_mean = mean;
      _statistics.per_bipartitioner[i].cut_variance = variance;
    }
    _statistics.best_cut = _best_cut;
    _statistics.best_feasible = _best_feasible;
    _statistics.best_imbalance = _best_imbalance;
    _statistics.best_bipartitioner = _best_bipartitioner;
  }

  void print_statistics() {
    std::size_t num_runs_total = 0;

    for (std::size_t i = 0; i < _bipartitioners.size(); ++i) {
      const auto &stats = _statistics.per_bipartitioner[i];
      const std::size_t num_runs = stats.num_feasible_partitions + stats.num_infeasible_partitions;
      num_runs_total += num_runs;

      LOG << logger::CYAN << "- " << _bipartitioner_names[i];
      LOG << logger::CYAN << "  * num=" << num_runs                            //
          << " num_feasible_partitions=" << stats.num_feasible_partitions      //
          << " num_infeasible_partitions=" << stats.num_infeasible_partitions; //
      LOG << logger::CYAN << "  * cut_mean=" << stats.cut_mean
          << " cut_variance=" << stats.cut_variance
          << " cut_std_dev=" << std::sqrt(stats.cut_variance);
    }

    LOG << logger::CYAN << "Winner: " << _bipartitioner_names[_best_bipartitioner];
    LOG << logger::CYAN << " * cut=" << _best_cut << " imbalance=" << _best_imbalance
        << " feasible=" << _best_feasible;
    LOG << logger::CYAN << "# of runs: " << num_runs_total << " of "
        << _bipartitioners.size() *
               std::clamp(_num_repetitions, _min_num_repetitions, _max_num_repetitions);
  }

  void run_bipartitioner(const std::size_t i) {
    PartitionedCSRGraph p_graph = _bipartitioners[i]->bipartition(
        std::move(_current_partition), std::move(_current_block_weights)
    );
    _refiner->refine(p_graph, *_p_ctx);

    const EdgeWeight current_cut = metrics::edge_cut_seq(p_graph);
    const double current_imbalance = metrics::imbalance(p_graph);
    const bool current_feasible = metrics::is_feasible(p_graph, *_p_ctx);

    _current_partition = p_graph.take_raw_partition();
    _current_block_weights = p_graph.take_raw_block_weights();

    // If the bipartition is feasible, track its stats
    if (current_feasible) {
      _statistics.per_bipartitioner[i].cuts.push_back(current_cut);
      ++_statistics.per_bipartitioner[i].num_feasible_partitions;
      _running_statistics[i].update(static_cast<double>(current_cut));
    } else {
      ++_statistics.per_bipartitioner[i].num_infeasible_partitions;
    };

    // Consider as best if it is feasible or the best partition is infeasible
    if (_best_feasible <= current_feasible &&
        (_best_feasible < current_feasible || current_cut < _best_cut ||
         (current_cut == _best_cut && current_imbalance < _best_imbalance))) {
      _best_cut = current_cut;
      _best_imbalance = current_imbalance;
      _best_feasible = current_feasible;

      // other _statistics.best_* are set during finalization
      _best_bipartitioner = i;

      std::swap(_current_partition, _best_partition);
      std::swap(_current_block_weights, _best_block_weights);
    }
  }

  const CSRGraph *_graph;
  const PartitionContext *_p_ctx;

  const InitialPartitioningContext &_i_ctx;
  std::size_t _min_num_repetitions;
  std::size_t _min_num_non_adaptive_repetitions;
  std::size_t _num_repetitions;
  std::size_t _max_num_repetitions;

  StaticArray<BlockID> _best_partition{0, static_array::small, static_array::seq};
  StaticArray<BlockID> _current_partition{0, static_array::small, static_array::seq};

  StaticArray<BlockWeight> _best_block_weights{2, 0, static_array::small, static_array::seq};
  StaticArray<BlockWeight> _current_block_weights{2, 0, static_array::small, static_array::seq};

  EdgeWeight _best_cut = std::numeric_limits<EdgeWeight>::max();
  bool _best_feasible = false;
  double _best_imbalance = 0.0;
  std::size_t _best_bipartitioner = 0;

  std::vector<std::string_view> _bipartitioner_names{};
  std::vector<std::unique_ptr<Bipartitioner>> _bipartitioners{};

  InitialRefiner *_refiner;

  std::vector<RunningVariance> _running_statistics{};
  Statistics _statistics{};
};
} // namespace kaminpar::shm::ip
