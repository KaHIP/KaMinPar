/*******************************************************************************
 * Functions to annotate the timer tree with aggregate timer information from
 * all PEs.
 *
 * @file:   timer.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#include "kaminpar-dist/timer.h"

#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

#include "kaminpar-mpi/wrapper.h"

namespace kaminpar::dist {
namespace {
class AlignedTable {
public:
  template <typename T> void update_next_column(const T &val) {
    const std::size_t len = arg_to_len(val);

    if (_current_column >= _column_len.size()) {
      _column_len.push_back(len);
    } else {
      _column_len[_current_column] = std::max(len, _column_len[_current_column]);
    }
    ++_current_column;
  }

  void next_row() {
    KASSERT(_current_column == _column_len.size());
    _current_column = 0;
  }

  template <typename T> std::string to_str_padded(const T &val, const bool pad_right = true) {
    const std::size_t padding_len = _column_len[_current_column] - arg_to_len(val);
    if (++_current_column == _column_len.size()) {
      _current_column = 0;
    }
    if (pad_right) {
      return arg_to_str(val) + std::string(padding_len, ' ');
    } else {
      return std::string(padding_len, ' ') + arg_to_str(val);
    }
  }

private:
  std::vector<std::size_t> _column_len;
  std::size_t _current_column = 0;

  template <typename T> static std::size_t arg_to_len(const T &val) {
    return arg_to_str(val).size();
  }

  template <typename T> static std::string arg_to_str(const T &val) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << val;
    return ss.str();
  }
};

template <typename Container> double compute_mean(const Container &vec) {
  const auto sum = static_cast<double>(std::accumulate(vec.begin(), vec.end(), 0.0));
  return sum / static_cast<double>(vec.size());
}

template <typename Container> double compute_sd(const Container &vec) {
  const double mean = compute_mean(vec);

  double sd_sum = 0;
  for (const auto &e : vec) {
    sd_sum += (e - mean) * (e - mean);
  }
  return std::sqrt(1.0 / static_cast<double>(vec.size()) * sd_sum);
}

struct NodeStatistics {
  PEID min_pe;
  double min;
  double mean;
  PEID max_pe;
  double max;
  double sd;

  std::vector<double> times;
};

template <std::size_t trunc_to = 1024, typename String>
std::vector<std::string> allgather_trunc_string(const String &str, MPI_Comm comm) {
  // copy str to char array
  char trunc[trunc_to];
  const std::size_t len = std::min(trunc_to - 1, str.length());
  str.copy(trunc, len);
  trunc[len] = 0;

  // collect on root
  const auto [size, rank] = mpi::get_comm_info(comm);
  std::vector<char> recv_buffer(size * trunc_to);
  mpi::allgather(trunc, trunc_to, recv_buffer.data(), trunc_to, comm);

  // build std::string objects on root
  std::vector<std::string> strings;
  for (PEID pe = 0; pe < size; ++pe) {
    strings.emplace_back(recv_buffer.data() + pe * trunc_to);
  }

  return strings;
}

template <std::size_t trunc_to = 1024, typename String>
std::vector<std::string> gather_trunc_string(const String &str, const PEID root, MPI_Comm comm) {
  // copy str to char array
  char trunc[trunc_to];
  const std::size_t len = std::min(trunc_to - 1, str.length());
  str.copy(trunc, len);
  trunc[len] = 0;

  // collect on root
  const auto [size, rank] = mpi::get_comm_info(comm);
  std::vector<char> recv_buffer(size * trunc_to);
  mpi::gather(trunc, trunc_to, recv_buffer.data(), trunc_to, root, comm);

  // build std::string objects on root
  std::vector<std::string> strings;
  if (rank == root) {
    for (PEID pe = 0; pe < size; ++pe) {
      strings.emplace_back(recv_buffer.data() + pe * trunc_to);
    }
  }

  return strings;
}

void generate_dummy_statistics(
    const Timer::TimerTreeNode &node, std::vector<NodeStatistics> &result
) {
  result.push_back({
      .min_pe = -1,
      .min = -1.0,
      .mean = -1.0,
      .max_pe = -1,
      .max = -1.0,
      .sd = -1.0,
      .times = {},
  });

  for (const auto &child : node.children) {
    generate_dummy_statistics(*child, result);
  }
}

void generate_statistics(
    const Timer::TimerTreeNode &node, std::vector<NodeStatistics> &result, MPI_Comm comm
) {
  const PEID root = 0;

  // Make sure that we are looking at the same timer node on each PE
  bool nondiverged_node = true;
  bool nondiverged_children = true;
  {
    constexpr std::size_t check_chars = 1024;

    const PEID rank = mpi::get_comm_rank(comm);

    // check that this timer node has the same number of children on each PE
    auto num_children = mpi::allgather(node.children.size(), comm);
    if (!std::all_of(num_children.begin(), num_children.end(), [&](const std::size_t num) {
          return num == node.children.size();
        })) {
      nondiverged_children = false;
    }

    auto names = allgather_trunc_string<check_chars>(node.name, comm);
    if (!std::all_of(names.begin(), names.end(), [&](const std::string &name) {
          return name.substr(0, check_chars) == node.name.substr(0, check_chars);
        })) {
      nondiverged_node = false;
    }

    auto descriptions = allgather_trunc_string<check_chars>(node.description, comm);
    if (!std::all_of(descriptions.begin(), descriptions.end(), [&](const std::string &description) {
          return description.substr(0, check_chars) == node.description.substr(0, check_chars);
        })) {
      nondiverged_node = false;
    }
  }

  if (nondiverged_node) {
    auto times = mpi::gather<double, std::vector<double>>(node.seconds(), 0, comm);
    const double mean = compute_mean(times);
    const double sd = compute_sd(times);

    if (mpi::get_comm_rank(comm) == root) {
      const auto min_it = std::min_element(times.begin(), times.end());
      const PEID min_pe = std::distance(times.begin(), min_it);
      const auto min = *min_it;

      const auto max_it = std::max_element(times.begin(), times.end());
      const PEID max_pe = std::distance(times.begin(), max_it);
      const auto max = *max_it;

      result.push_back({
          .min_pe = min_pe,
          .min = min,
          .mean = mean,
          .max_pe = max_pe,
          .max = max,
          .sd = sd,
          .times = std::move(times),
      });
    }

    if (nondiverged_children) {
      for (const auto &child : node.children) {
        generate_statistics(*child, result, comm);
      }
    } else {
      for (const auto &child : node.children) {
        generate_dummy_statistics(*child, result);
      }
    }
  } else {
    generate_dummy_statistics(node, result);
  }
}

AlignedTable align_statistics(const std::vector<NodeStatistics> &statistics) {
  AlignedTable table;
  for (auto &entry : statistics) {
    table.update_next_column(entry.min_pe);
    table.update_next_column(entry.min);
    table.update_next_column(entry.mean);
    table.update_next_column(entry.max_pe);
    table.update_next_column(entry.max);
    // table.update_next_column(entry.sd);
    table.next_row();
  }
  return table;
}

void annotate_timer_tree(
    Timer::TimerTreeNode &node,
    std::size_t &pos,
    const std::vector<NodeStatistics> &statistics,
    AlignedTable &table
) {
  const auto &entry = statistics[pos++];

  std::stringstream ss;
  if (entry.min >= 0) {
    ss << "[" << table.to_str_padded(entry.min_pe, false) << " : " << table.to_str_padded(entry.min)
       << "s | " << table.to_str_padded(entry.mean) << "s | "
       << table.to_str_padded(entry.max_pe, false) << " : " << table.to_str_padded(entry.max)
       << "s]";
  } else {
    ss << "N/A ";
  }

  // Also print running times that deviate by more than 3 SDs
  // Disable: produces too much output on large PE counts
  /*
  for (std::size_t pe = 0; pe < entry.times.size(); ++pe) {
      const double t = entry.times[pe];
      if (t < entry.mean - 3 * entry.sd || entry.mean + 3 * entry.sd < t) {
          ss << pe << "/" << t << " s ";
      }
  }
  */
  node.annotation = ss.str();

  for (const auto &child : node.children) {
    annotate_timer_tree(*child, pos, statistics, table);
  }
}
} // namespace

void finalize_distributed_timer(Timer &timer, MPI_Comm comm) {
  std::vector<NodeStatistics> statistics;
  generate_statistics(timer.tree(), statistics, comm);
  if (mpi::get_comm_rank(comm) == 0) {
    AlignedTable table = align_statistics(statistics);
    table.update_next_column("PE");
    table.update_next_column("min");
    table.update_next_column("avg");
    table.update_next_column("PE");
    table.update_next_column("max");
    table.next_row();

    // add captions
    std::stringstream ss;
    ss << " " << table.to_str_padded("PE", false) << " : " << table.to_str_padded("min") << "    "
       << table.to_str_padded("avg") << "    " << table.to_str_padded("PE", false) << " : "
       << table.to_str_padded("max") << "  ";
    timer.annotate(ss.str());

    std::size_t pos = 0;
    annotate_timer_tree(timer.tree(), pos, statistics, table);
  }
}
} // namespace kaminpar::dist
