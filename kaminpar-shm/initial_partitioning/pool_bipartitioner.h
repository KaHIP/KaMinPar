/*******************************************************************************
 * @file:   pool_bipartitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Initial partitioner that uses a portfolio of initial partitioning
 * algorithms. Each graphutils is repeated multiple times. Algorithms that are
 * unlikely to beat the best partition found so far are executed less often
 * than promising candidates.
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/factories.h"
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
  struct MemoryContext {
    GreedyGraphGrowingBipartitioner::MemoryContext ggg_m_ctx;
    bfs::BfsBipartitionerBase::MemoryContext bfs_m_ctx;
    RandomBipartitioner::MemoryContext rand_m_ctx;
    InitialRefiner::MemoryContext ref_m_ctx;

    std::size_t memory_in_kb() const {
      return ggg_m_ctx.memory_in_kb() + bfs_m_ctx.memory_in_kb() + rand_m_ctx.memory_in_kb() +
             ref_m_ctx.memory_in_kb();
    }
  };

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

  PoolBipartitioner(
      const Graph &graph,
      const PartitionContext &p_ctx,
      const InitialPartitioningContext &i_ctx,
      MemoryContext m_ctx = {}
  )
      : _graph(graph),
        _p_ctx(p_ctx),
        _i_ctx(i_ctx),
        _min_num_repetitions(i_ctx.min_num_repetitions),
        _min_num_non_adaptive_repetitions(i_ctx.min_num_non_adaptive_repetitions),
        _max_num_repetitions(i_ctx.max_num_repetitions),
        _m_ctx(std::move(m_ctx)),
        _refiner(
            create_initial_refiner(_graph, _p_ctx, _i_ctx.refinement, std::move(_m_ctx.ref_m_ctx))
        ) {
    _refiner->initialize(_graph);
  }

  template <typename BipartitionerType, typename... BipartitionerArgs>
  void register_bipartitioner(const std::string &name, BipartitionerArgs &&...args) {
    KASSERT(
        std::find(_bipartitioner_names.begin(), _bipartitioner_names.end(), name) ==
        _bipartitioner_names.end()
    );
    auto *instance =
        new BipartitionerType(_graph, _p_ctx, _i_ctx, std::forward<BipartitionerArgs>(args)...);
    _bipartitioners.push_back(std::unique_ptr<BipartitionerType>(instance));
    _bipartitioner_names.push_back(name);
    _running_statistics.emplace_back();
    _statistics.per_bipartitioner.emplace_back();
  }

  const std::string &bipartitioner_name(const std::size_t i) {
    return _bipartitioner_names[i];
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
    _best_partition = StaticArray<BlockID>(_graph.n());
  }

  PartitionedGraph bipartition() {
    KASSERT(_current_partition.size() >= _graph.n());
    KASSERT(_best_partition.size() >= _graph.n());

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

    return {_graph, 2, std::move(_best_partition)};
  }

  MemoryContext free() {
    _m_ctx.ref_m_ctx = _refiner->free();
    return std::move(_m_ctx);
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
    DBG << "Running bipartitioner " << _bipartitioner_names[i] << " on graph with n=" << _graph.n()
        << " m=" << _graph.m();
    PartitionedGraph p_graph = _bipartitioners[i]->bipartition(std::move(_current_partition));
    DBG << " -> running refiner ...";
    _refiner->refine(p_graph, _p_ctx);
    DBG << " -> cut=" << metrics::edge_cut(p_graph) << " imbalance=" << metrics::imbalance(p_graph);

    const EdgeWeight current_cut = metrics::edge_cut_seq(p_graph);
    const double current_imbalance = metrics::imbalance(p_graph);
    const bool current_feasible = metrics::is_feasible(p_graph, _p_ctx);
    _current_partition = p_graph.take_raw_partition();

    // record statistics if the bipartition is feasible
    if (current_feasible) {
      _statistics.per_bipartitioner[i].cuts.push_back(current_cut);
      ++_statistics.per_bipartitioner[i].num_feasible_partitions;
      _running_statistics[i].update(static_cast<double>(current_cut));
    } else {
      ++_statistics.per_bipartitioner[i].num_infeasible_partitions;
    };

    // consider as best if it is feasible or the best partition is infeasible
    if (_best_feasible <= current_feasible &&
        (_best_feasible < current_feasible || current_cut < _best_cut ||
         (current_cut == _best_cut && current_imbalance < _best_imbalance))) {
      _best_cut = current_cut;
      _best_imbalance = current_imbalance;
      _best_feasible = current_feasible;
      _best_bipartitioner = i; // other _statistics.best_* are set during finalization
      std::swap(_current_partition, _best_partition);
    }
  }

  const Graph &_graph;
  const PartitionContext &_p_ctx;
  const InitialPartitioningContext &_i_ctx;
  std::size_t _min_num_repetitions;
  std::size_t _min_num_non_adaptive_repetitions;
  std::size_t _num_repetitions;
  std::size_t _max_num_repetitions;

  MemoryContext _m_ctx{};

  StaticArray<BlockID> _best_partition{_graph.n()};
  EdgeWeight _best_cut{std::numeric_limits<EdgeWeight>::max()};
  bool _best_feasible{false};
  double _best_imbalance{0.0};
  std::size_t _best_bipartitioner{0};
  StaticArray<BlockID> _current_partition{_graph.n()};

  std::vector<std::string> _bipartitioner_names{};
  std::vector<std::unique_ptr<Bipartitioner>> _bipartitioners{};
  std::unique_ptr<InitialRefiner> _refiner;

  std::vector<RunningVariance> _running_statistics{};
  Statistics _statistics{};
};

/**
 * Factory for PoolBipartitioner.
 * Lifetime of the instance returned by create() is bound to this object.
 * Create frees the previously created bipartitioner instance.
 * Hence, there can only be one PoolBipartitioner per factory at a time.
 */
class PoolBipartitionerFactory {
public:
  std::unique_ptr<PoolBipartitioner> create(
      const Graph &graph,
      const PartitionContext &p_ctx,
      const InitialPartitioningContext &i_ctx,
      PoolBipartitioner::MemoryContext m_ctx = {}
  ) {
    auto pool = std::make_unique<PoolBipartitioner>(graph, p_ctx, i_ctx, std::move(m_ctx));
    pool->register_bipartitioner<GreedyGraphGrowingBipartitioner>(
        "greedy_graph_growing", pool->_m_ctx.ggg_m_ctx
    );
    pool->register_bipartitioner<AlternatingBfsBipartitioner>(
        "bfs_alternating", pool->_m_ctx.bfs_m_ctx
    );
    pool->register_bipartitioner<LighterBlockBfsBipartitioner>(
        "bfs_lighter_block", pool->_m_ctx.bfs_m_ctx
    );
    pool->register_bipartitioner<LongerQueueBfsBipartitioner>(
        "bfs_longer_queue", pool->_m_ctx.bfs_m_ctx
    );
    pool->register_bipartitioner<ShorterQueueBfsBipartitioner>(
        "bfs_shorter_queue", pool->_m_ctx.bfs_m_ctx
    );
    pool->register_bipartitioner<SequentialBfsBipartitioner>(
        "bfs_sequential", pool->_m_ctx.bfs_m_ctx
    );
    pool->register_bipartitioner<RandomBipartitioner>("random", pool->_m_ctx.rand_m_ctx);
    return pool;
  }
};
} // namespace kaminpar::shm::ip
