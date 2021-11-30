/*******************************************************************************
 * @file:   distributed_timer.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#include "dkaminpar/utility/distributed_timer.h"

#include "dkaminpar/mpi_wrapper.h"

#include <cmath>
#include <numeric>
#include <sstream>

namespace dkaminpar::timer {
using shm::Timer;

namespace {
class AlignedTable {
public:
  template<typename T>
  void update_next_column(const T &val) {
    const std::size_t len = arg_to_len(val);

    if (_current_column >= _column_len.size()) {
      _column_len.push_back(len);
    } else {
      _column_len[_current_column] = std::max(len, _column_len[_current_column]);
    }
    ++_current_column;
  }

  void next_row() {
    ASSERT(_current_column == _column_len.size());
    _current_column = 0;
  }

  template<typename T>
  std::string to_str_padded(const T &val) {
    const std::size_t padding_len = _column_len[_current_column] - arg_to_len(val);
    if (++_current_column == _column_len.size()) { _current_column = 0; }
    return arg_to_str(val) + std::string(padding_len, ' ');
  }

private:
  std::vector<std::size_t> _column_len;
  std::size_t _current_column = 0;

  template<typename T>
  static std::size_t arg_to_len(const T &val) {
    return arg_to_str(val).size();
  }

  template<typename T>
  static std::string arg_to_str(const T &val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
  }
};

double compute_mean(const auto &vec) {
  const auto sum = static_cast<double>(std::accumulate(vec.begin(), vec.end(), 0.0));
  return sum / static_cast<double>(vec.size());
}

double compute_sd(const auto &vec) {
  const double mean = compute_mean(vec);

  double sd_sum = 0;
  for (const auto &e : vec) { sd_sum += (e - mean) * (e - mean); }
  return std::sqrt(1.0 / static_cast<double>(vec.size()) * sd_sum);
}

struct NodeStatistics {
  double min;
  double mean;
  double max;
  double sd;

  std::vector<double> times;
};

void generate_statistics(const Timer::TimerTreeNode &node, MPI_Comm comm, std::vector<NodeStatistics> &result) {
  const auto times = mpi::gather<double, std::vector>(node.seconds(), 0, comm);

  if (mpi::get_comm_rank(comm) == 0) {
    const auto [min, max] = std::ranges::minmax(times);
    result.push_back({
        .min = min,
        .mean = compute_mean(times),
        .max = max,
        .sd = compute_sd(times),
        .times = std::move(times),
    });
  }

  for (const auto &child : node.children) { generate_statistics(*child, comm, result); }
}

AlignedTable align_statistics(const std::vector<NodeStatistics> &statistics) {
  AlignedTable table;
  for (auto &entry : statistics) {
    table.update_next_column(entry.min);
    table.update_next_column(entry.mean);
    table.update_next_column(entry.max);
    table.update_next_column(entry.sd);
    table.next_row();
  }
  return table;
}

void annotate_timer_tree(Timer::TimerTreeNode &node, std::size_t pos, const std::vector<NodeStatistics> &statistics,
                         AlignedTable &table) {
  const auto &entry = statistics[pos];

  std::stringstream ss;
  ss << "|" << table.to_str_padded(entry.min) << " s <= " << table.to_str_padded(entry.mean)
     << " s <= " << table.to_str_padded(entry.max) << " s :: sd=" << table.to_str_padded(entry.sd) << " s| ";

  // also print list of outliners
  for (std::size_t pe = 0; pe < entry.times.size(); ++pe) {
    const double t = entry.times[pe];
    if (t < entry.mean - 2 * entry.sd || entry.mean + 2 * entry.sd < t) { ss << pe << "/" << t << " s "; }
  }
  node.annotation = ss.str();

  for (const auto &child : node.children) { annotate_timer_tree(*child, ++pos, statistics, table); }
}
} // namespace

void finalize_distributed_timer(shm::Timer &timer, MPI_Comm comm) {
  std::vector<NodeStatistics> statistics;
  generate_statistics(timer.tree(), comm, statistics);
  if (mpi::get_comm_rank(comm) == 0) {
    AlignedTable table = align_statistics(statistics);
    annotate_timer_tree(timer.tree(), 0, statistics, table);
  }
}
} // namespace dkaminpar::timer