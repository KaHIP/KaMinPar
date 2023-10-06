/*******************************************************************************
 * Hierarchical timer for easy time measurements.
 *
 * @file:   timer.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-common/timer.h"

#include <iomanip>
#include <sstream>

using namespace std::literals;

namespace kaminpar {
namespace {
[[nodiscard]] std::string string_make_machine_readable(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), [](const auto &ch) {
    return std::tolower(ch);
  });
  std::replace(str.begin(), str.end(), ' ', '_');
  return str;
}

template <typename Value> [[nodiscard]] std::size_t get_printed_length(const Value value) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(3) << value;
  return ss.str().size();
}
} // namespace

std::string Timer::TimerTreeNode::build_display_name_mr() const {
  std::stringstream ss;
  ss << string_make_machine_readable(std::string(name.begin(), name.end()));
  if (!description.empty()) {
    ss << "[" << string_make_machine_readable(description) << "]";
  }
  return ss.str();
}

std::string Timer::TimerTreeNode::build_display_name_hr() const {
  std::stringstream ss;
  ss << name;
  if (!description.empty()) {
    ss << " (" << description << ")";
  }
  return ss.str();
}

Timer::Timer(std::string_view name) : _name{name} {
  _tree.root.start = timer::now();
}

Timer &Timer::global() {
  static Timer timer("Global Timer");
  return timer;
}

void __attribute__((noinline)) Timer::start_timer_impl() {
  asm volatile("" ::: "memory");
  _tree.current->start = std::chrono::high_resolution_clock::now();
}

void __attribute__((noinline)) Timer::stop_timer_impl() {
  asm volatile("" ::: "memory");
  const TimePoint end = std::chrono::high_resolution_clock::now();
  _tree.current->elapsed += end - _tree.current->start;
}

void Timer::reset() {
  _tree = TimerTree{};
  _tree.current = &_tree.root;
  _tree.root.start = timer::now();
  _disabled = 0;
}

//
// Machine-readable output
//

void Timer::print_machine_readable(std::ostream &out, const int max_depth) {
  for (const auto &node : _tree.root.children) {
    print_node_mr(out, "", node.get(), max_depth);
  }
  out << "\n";
}

void Timer::print_node_mr(
    std::ostream &out, const std::string &prefix, const TimerTreeNode *node, const int max_depth
) const {
  if (max_depth < 0) {
    return;
  }

  // Print this node
  const std::string display_name = prefix + node->build_display_name_mr();
  out << display_name << "=" << node->seconds() << " ";

  // Print children
  const std::string child_prefix = display_name + ".";
  for (const auto &child : node->children) {
    print_node_mr(out, child_prefix, child.get(), max_depth - 1);
  }
}

//
// Human-readable output
//

void Timer::print_human_readable(std::ostream &out, const int max_depth) {
  if (max_depth < 0) {
    return;
  }

  _hr_time_col = std::max(_name.size() + kNameDel.size(), compute_time_col(0, &_tree.root));
  _hr_max_time_len = compute_time_len(&_tree.root);
  _hr_max_restarts_len = compute_restarts_len(&_tree.root);
  out << _name;
  print_padded_timing(out, _name.size(), &_tree.root);
  if (!_annotation.empty()) {
    out << std::string(kSpaceBetweenRestartsAndAnnotation, ' ') << _annotation;
  }
  out << std::endl;

  print_children_hr(out, "", &_tree.root, max_depth - 1);
}

void Timer::print_children_hr(
    std::ostream &out,
    const std::string &base_prefix,
    const TimerTreeNode *node,
    const int max_depth
) const {
  if (max_depth < 0) {
    return;
  }

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
    print_padded_timing(out, prefix.size() + display_name.size(), child.get());
    if (!child->annotation.empty()) {
      out << std::string(kSpaceBetweenRestartsAndAnnotation, ' ') << child->annotation;
    }
    out << std::endl;
    print_children_hr(out, child_prefix, child.get(), max_depth - 1);
  }
}

void Timer::print_padded_timing(
    std::ostream &out, const std::size_t start_col, const TimerTreeNode *node
) const {
  using namespace std::literals;

  // print this node
  const std::size_t time_padding_len = _hr_time_col - start_col - kNameDel.size();
  std::string time_padding =
      time_padding_len > 0 ? std::string(time_padding_len - 1, kPadding) + ' ' : "";
  const double time = node->seconds();
  out << kNameDel << time_padding << std::fixed << std::setprecision(3) << time << kSecondsUnit;

  const std::size_t time_len = get_printed_length(time);
  const std::size_t restarts_padding_length =
      _hr_max_time_len - time_len + kSpaceBetweenTimeAndRestarts;
  const std::size_t tail_padding_length = _hr_max_restarts_len - get_printed_length(node->restarts);
  out << std::string(restarts_padding_length, ' ');

  if (node->restarts > 1) {
    out << "(" << node->restarts << ")" << std::string(tail_padding_length, ' ');
  } else if (_hr_max_restarts_len > 0) {               // otherwise, there are no restarts and
                                                       // we are already aligned
    out << std::string(2 + _hr_max_restarts_len, ' '); // +2 for (, )
  }
}

[[nodiscard]] std::size_t
Timer::compute_time_col(const std::size_t parent_prefix_len, const TimerTreeNode *node) const {
  using namespace std::literals;
  const std::size_t prefix_len = (node->parent == nullptr)
                                     ? 0                                       // root
                                     : parent_prefix_len + kTailBranch.size(); // inner node or leaf

  std::size_t col = prefix_len + node->build_display_name_hr().size() + kNameDel.size();
  for (const auto &child : node->children) {
    col = std::max(col, compute_time_col(prefix_len, child.get()));
  }

  return col;
}

[[nodiscard]] std::size_t Timer::compute_time_len(const TimerTreeNode *node) const {
  std::size_t space = get_printed_length(node->seconds());
  for (const auto &child : node->children) {
    space = std::max(space, compute_time_len(child.get()));
  }
  return space;
}

[[nodiscard]] std::size_t Timer::compute_restarts_len(const TimerTreeNode *node) const {
  std::size_t space = node->restarts > 1 ? get_printed_length(node->restarts) : 0;
  for (const auto &child : node->children) {
    space = std::max(space, compute_restarts_len(child.get()));
  }
  return space;
}
} // namespace kaminpar
