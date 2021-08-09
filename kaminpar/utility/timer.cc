/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
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
  return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()) / 1000.0;
}

[[nodiscard]] std::size_t get_printed_length(const auto value) {
  std::stringstream ss;
  ss << value;
  return ss.str().size();
}
} // namespace

std::string Timer::TimerTreeNode::build_display_name_mr() const {
  std::stringstream ss;
  ss << string_make_machine_readable(name.data());
  if (!description.empty()) { ss << "_" << string_make_machine_readable(description); }
  return ss.str();
}

std::string Timer::TimerTreeNode::build_display_name_hr() const {
  std::stringstream ss;
  ss << name.data();
  if (!description.empty()) { ss << " " << description; }
  return ss.str();
}

Timer::Timer(std::string_view name) : _name{name} {
  _enabled[Type::DEFAULT] = true;
  _tree.root.start = timer::now();
}

Timer &Timer::global() {
  static Timer timer{"Global Timer"};
  return timer;
}

//
// Machine-readable output
//

void Timer::print_machine_readable(std::ostream &out) {
  for (const auto &node : _tree.root.children) { print_node_mr(out, "", node.get()); }
  out << "\n";
}

void Timer::print_node_mr(std::ostream &out, const std::string &prefix, const TimerTreeNode *node) {
  const std::string display_name = prefix + node->build_display_name_mr();

  // print this node
  out << display_name << "=" << to_seconds(node->elapsed) << " ";

  // print children
  const std::string child_prefix = display_name + ".";
  for (const auto &child : node->children) { print_node_mr(out, child_prefix, child.get()); }
}

//
// Human-readable output
//

void Timer::print_human_readable(std::ostream &out) {
  const std::size_t time_col = std::max(_name.size() + kNameDel.size(), compute_time_col(0, &_tree.root));
  const std::size_t max_time_len = compute_time_len(&_tree.root);
  out << _name;
  print_padded_timing(out, _name.size(), time_col, max_time_len, &_tree.root);
  out << std::endl;
  print_children_hr(out, "", &_tree.root, time_col, max_time_len);
}

void Timer::print_children_hr(std::ostream &out, const std::string &base_prefix, const TimerTreeNode *node,
                                         const std::size_t time_col, const std::size_t max_time_len) const {
  const std::string prefix_mid = base_prefix + std::string(kBranch);
  const std::string child_prefix_mid = base_prefix + std::string(kEdge);
  const std::string prefix_end = base_prefix + std::string(kTailBranch);
  const std::string child_prefix_end = base_prefix + std::string(kTailEdge);

  for (const auto &child : node->children) {
    const bool is_last = (child == node->children.back());
    const auto &prefix = is_last ? prefix_end : prefix_mid;
    const auto &child_prefix = is_last ? child_prefix_end : child_prefix_mid;

    const std::string display_name = child->build_display_name_hr();
    out << prefix << display_name;
    print_padded_timing(out, prefix.size() + display_name.size(), time_col, max_time_len, child.get());
    out << std::endl;
    print_children_hr(out, child_prefix, child.get(), time_col, max_time_len);
  }
}

void Timer::print_padded_timing(std::ostream &out, const std::size_t start_col, const std::size_t time_col,
                                const std::size_t max_time_len, const TimerTreeNode *node) {
  using namespace std::literals;

  // print this node
  const std::size_t time_padding_len = time_col - start_col - kNameDel.size();
  std::string time_padding = time_padding_len > 0 ? std::string(time_padding_len - 1, kPadding) + ' ' : "";
  const double time = to_seconds(node->elapsed);
  out << kNameDel << time_padding << time << kSecondsUnit;

  if (node->restarts > 1) {
    const std::size_t time_len = get_printed_length(time);
    const std::size_t restarts_padding_length = max_time_len - time_len + kSpaceBetweenTimeAndRestarts;
    out << std::string(restarts_padding_length, ' ') << "(" << node->restarts << ")";
  }
}

[[nodiscard]] std::size_t Timer::compute_time_col(const std::size_t parent_prefix_len,
                                                  const TimerTreeNode *node) const {
  using namespace std::literals;
  const std::size_t prefix_len = (node->parent == nullptr)
                                     ? 0                                       // root
                                     : parent_prefix_len + kTailBranch.size(); // inner node or leaf

  std::size_t col = prefix_len + node->build_display_name_hr().size() + kNameDel.size();
  for (const auto &child : node->children) { col = std::max(col, compute_time_col(prefix_len, child.get())); }

  return col;
}

[[nodiscard]] std::size_t Timer::compute_time_len(const TimerTreeNode *process) const {
  std::size_t space = get_printed_length(to_seconds(process->elapsed));
  for (const auto &sub_subtree : process->children) {
    space = std::max(space, compute_time_len(sub_subtree.get()));
  }
  return space;
}
} // namespace kaminpar