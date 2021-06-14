/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2020 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#include "utility/timer.h"

#include <sstream>

using namespace std::literals;

namespace kaminpar {
namespace {
[[nodiscard]] std::string string_make_machine_readable(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), [](const auto &ch) { return std::tolower(ch); });
  std::replace(str.begin(), str.end(), ' ', '_');
  return str;
}

[[nodiscard]] double to_seconds(const Timer::Duration &duration) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0;
}

[[nodiscard]] std::size_t get_printed_double_length(const double value) {
  std::stringstream ss;
  ss << value;
  return ss.str().size();
}

[[nodiscard]] std::string create_padding(const std::size_t length, const std::size_t margin, const char filler) {
  return (length > 2 * margin)
             ? std::string(margin, ' ') + std::string(length - 2 * margin, filler) + std::string(margin, ' ')
             : std::string(length, ' ');
}
} // namespace

Timer::Timer(const std::string &name) : _name{name} { _global_root.root.start = timer::now(); }

Timer &Timer::global() {
  static Timer timer("Global Timer");
  return timer;
}

void Timer::print_machine_readable(std::ostream &out) {
  for (const auto &[name, subtree] : _global_root.root.children) {
    print_subprocess_mr(out, "", name, subtree.get(), false);
  }
  for (const auto &[name, subtree] : _global_root.root.local_children) {
    print_subprocess_mr(out, "", name, subtree.get(), true);
  }
  out << "\n";
}

void Timer::print_subprocess_mr(std::ostream &out, const std::string &prefix, const std::string &name,
                                const TimerTreeNode *subtree, const bool is_local) {
  const std::string full_name = prefix + string_make_machine_readable(name);
  const double time = to_seconds(subtree->elapsed);
  out << (is_local ? "L." : "G.") << full_name << "=" << time << " ";
  for (const auto &[sub_name, sub_subtree] : subtree->children) {
    print_subprocess_mr(out, full_name + ".", sub_name, sub_subtree.get(), false);
  }
  for (const auto &[sub_name, sub_subtree] : subtree->local_children) {
    print_subprocess_mr(out, full_name + ".", sub_name, sub_subtree.get(), true);
  }
}

void Timer::print_human_readable(std::ostream &out) {
  const std::size_t time_column = compute_time_output_column(0, _name, &_global_root.root);
  const std::size_t time_space = compute_max_time_space(&_global_root.root);
  out << "G " << _name;
  print_padded_timing(out, _name.size(), time_column, time_space, &_global_root.root);
  out << std::endl;
  print_subprocess_children_hr(out, "", &_global_root.root, time_column, time_space);
}

void Timer::print_padded_timing(std::ostream &out, const std::size_t start_column, const std::size_t time_column,
                                const std::size_t time_space, const TimerTreeNode *of_process) {
  using namespace std::literals;
  const std::size_t time_padding_length = time_column - start_column - ":"s.size();
  const std::string time_padding = create_padding(time_padding_length, 1, '.');
  const double time = to_seconds(of_process->elapsed);
  out << ":" << time_padding << time << " s";
  if (of_process->restarts > 1) {
    const std::size_t time_length = get_printed_double_length(time);
    const std::size_t restarts_padding_length = time_space - time_length + kSpaceBetweenTimeAndRestarts;
    out << create_padding(restarts_padding_length, 0, ' ') << "(" << of_process->restarts << ")";
  }
}

void Timer::print_subprocess_children_hr(std::ostream &out, const std::string &base_prefix,
                                         const TimerTreeNode *process, const std::size_t time_column,
                                         const std::size_t time_space) const {
  const std::string prefix_mid = base_prefix + "|-- ";
  const std::string child_prefix_mid = base_prefix + "|   ";
  const std::string prefix_end = base_prefix + "`-- ";
  const std::string child_prefix_end = base_prefix + "    ";

  auto print_children = [&](const auto &children, const bool is_local) {
    for (const auto &[sub_name, sub_subtree] : children) {
      const bool is_last = (sub_name == children.rbegin()->first);
      const auto &prefix = is_last ? prefix_end : prefix_mid;
      const auto &child_prefix = is_last ? child_prefix_end : child_prefix_mid;

      out << (is_local ? "L " : "G ");
      out << prefix << sub_name;
      print_padded_timing(out, prefix.size() + sub_name.size(), time_column, time_space, sub_subtree.get());
      out << std::endl;
      print_subprocess_children_hr(out, child_prefix, sub_subtree.get(), time_column, time_space);
    }
  };

  print_children(process->children, false);
  print_children(process->local_children, true);
}
[[nodiscard]] std::size_t Timer::compute_time_output_column(std::size_t prefix_length, const std::string &name,
                                                            const TimerTreeNode *process) const {
  using namespace std::literals;
  const std::size_t new_prefix_length = (process->parent == nullptr) ? ""s.size() : prefix_length + "`-- "s.size();
  std::size_t column = new_prefix_length + name.size() + ": "s.size();

  auto update_with_max_length = [&](const auto &children) {
    for (const auto &[sub_name, sub_subtree] : children) {
      column = std::max(column, compute_time_output_column(new_prefix_length, sub_name, sub_subtree.get()));
    }
  };

  update_with_max_length(process->children);
  update_with_max_length(process->local_children);

  return column;
}

[[nodiscard]] std::size_t Timer::compute_max_time_space(const TimerTreeNode *process) const {
  std::size_t space = get_printed_double_length(to_seconds(process->elapsed));
  for (const auto &[sub_name, sub_subtree] : process->children) {
    space = std::max(space, compute_max_time_space(sub_subtree.get()));
  }
  return space;
}

timer::FlatTimer &timer::FlatTimer::global() {
  static FlatTimer instance{};
  return instance;
}
} // namespace kaminpar