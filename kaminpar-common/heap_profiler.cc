/*******************************************************************************
 * Heap profiler to measure heap memory usage.
 *
 * @file:   heap_profiler.cc
 * @author: Daniel Salwasser
 * @date:   21.10.2023
 ******************************************************************************/
#include "kaminpar-common/heap_profiler.h"

namespace kaminpar::heap_profiler {

template <class T> T *NoProfilAllocator<T>::allocate(const size_t n) const {
  if (n == 0) {
    return nullptr;
  }

  if (n > static_cast<size_t>(-1) / sizeof(T)) {
    throw std::bad_array_new_length();
  }

#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  void *const pv = std_malloc(n * sizeof(T));
#else
  void *const pv = std::malloc(n * sizeof(T));
#endif
  if (!pv) {
    throw std::bad_alloc();
  }

  return static_cast<T *>(pv);
}

template <class T> void NoProfilAllocator<T>::deallocate(T *const p, size_t) const noexcept {
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  std_free(p);
#else
  std::free(p);
#endif
}

template <class T> T *NoProfilAllocator<T>::construct() const {
  T *t = allocate(1);
  new (t) T();
  return t;
}

template <class T> void NoProfilAllocator<T>::destruct(T *const t) const {
  t->~T();
  deallocate(t, 1);
}

HeapProfiler &HeapProfiler::global() {
  static HeapProfiler global{"Global Heap Profiler"};
  return global;
}

HeapProfiler::HeapProfiler(std::string_view name) : _name(name) {
  _tree.root.name = name;
  _tree.root.parent = &_tree.root;
}

HeapProfiler::~HeapProfiler() {
  _tree.root.free(_node_allocator);
}

void HeapProfiler::enable() {
  _enabled = true;
}

void HeapProfiler::disable() {
  _enabled = false;
}

void HeapProfiler::start_profile(std::string_view name, std::string description) {
  auto &children = _tree.currentNode->children;

  if (children.find(name) == children.end()) {
    HeapProfileTreeNode *node = _node_allocator.construct();
    node->name = name;
    node->description = description;
    node->parent = _tree.currentNode;

    _tree.currentNode->ordered_children.push_back(node);
    children[name] = node;

    _tree.currentNode = node;
  } else {
    _tree.currentNode = children[name];
  }
}

void HeapProfiler::stop_profile() {
  auto *current = _tree.currentNode;
  auto *parent = current->parent;

  parent->allocs += current->allocs;
  parent->frees += current->frees;
  parent->alloc_size += current->alloc_size;

  _tree.currentNode = parent;
}

ScopedHeapProfiler
HeapProfiler::start_scoped_profile(std::string_view name, std::string description) {
  return ScopedHeapProfiler{name, description};
}

void HeapProfiler::record_alloc(const void *ptr, std::size_t size) {
  if (_enabled) {
    _tree.currentNode->allocs++;
    _tree.currentNode->alloc_size += size;
    _address_map[ptr] = size;

    _allocs++;
    _total_alloc += size;
    if (std::size_t _current_alloc = _total_alloc - _total_free; _current_alloc > _max_alloc) {
      _max_alloc = _current_alloc;
    }
  }
}

void HeapProfiler::record_free(const void *ptr) {
  if (_enabled) {
    _tree.currentNode->frees++;

    _frees++;
    _total_free += _address_map[ptr];
    _address_map.erase(ptr);
  }
}

void HeapProfiler::print_heap_profile(std::ostream &out) {
  HeapProfileTreeNode &root = *_tree.currentNode;
  HeapProfileTreeStats stats = calculate_stats(root);

  stats.max_alloc = std::max(kAllocSizeTitle.length(), to_megabytes(stats.max_alloc).length());
  stats.max_allocs = std::max(kAllocsTitle.length(), std::to_string(stats.max_allocs).length());
  stats.max_frees = std::max(kFreesTitle.length(), std::to_string(stats.max_frees).length());

  out << "Max Memory Usage: " << to_megabytes(_max_alloc) << " (mb)" << '\n';

  out << std::string(stats.max_len + 2 + 10, '-') << ' ';
  out << kAllocSizeTitle << " " << std::string(stats.max_alloc - kAllocSizeTitle.length(), ' ');
  out << kAllocsTitle << " " << std::string(stats.max_allocs - kAllocsTitle.length(), ' ');
  out << kFreesTitle << " " << std::string(stats.max_frees - kFreesTitle.length(), ' ') << '\n';

  print_heap_tree_node(out, root, stats);
  out << '\n';
}

std::size_t HeapProfiler::get_max_alloc() {
  return _max_alloc;
}

std::size_t HeapProfiler::get_alloc() {
  return _tree.currentNode->alloc_size;
}

std::size_t HeapProfiler::get_allocs() {
  return _tree.currentNode->allocs;
}

std::size_t HeapProfiler::get_frees() {
  return _tree.currentNode->frees;
}

HeapProfileTreeStats HeapProfiler::calculate_stats(const HeapProfileTreeNode &node) {
  std::size_t name_length = node.name.length();
  if (!node.description.empty()) {
    name_length += node.description.length() + 2;
  }

  HeapProfileTreeStats stats = {name_length, node.alloc_size, node.allocs, node.frees};

  for (auto const &child : node.ordered_children) {
    HeapProfileTreeStats child_stats = calculate_stats(*child);
    stats.max_len = std::max(stats.max_len, child_stats.max_len + kBranchLength);
    stats.max_alloc = std::max(stats.max_alloc, child_stats.max_alloc);
    stats.max_allocs = std::max(stats.max_allocs, child_stats.max_allocs);
    stats.max_frees = std::max(stats.max_frees, child_stats.max_frees);
  }

  return stats;
}

void HeapProfiler::print_heap_tree_node(
    std::ostream &out,
    const HeapProfileTreeNode &node,
    HeapProfileTreeStats stats,
    std::size_t depth,
    bool last
) {
  if (depth > 0) {
    std::size_t leading_whitespaces = (depth - 1) * kBranchLength;
    out << std::string(leading_whitespaces, ' ') << (last ? kTailBranch : kBranch);
  }

  float percentage =
      (node.parent->alloc_size == 0) ? 1 : (node.alloc_size / (float)node.parent->alloc_size);
  out << "(";
  if (percentage >= 0.999995) {
    out << "100.00";
  } else {
    if (percentage < 0.1) {
      out << "0";
    }
    out << percentage * 100;
  }
  out << "%) " << node.name;

  std::size_t padding_length = stats.max_len - (depth * kBranchLength + node.name.length());
  if (!node.description.empty()) {
    padding_length -= node.description.length() + 2;
    out << "(" << node.description << ")";
  }

  out << kNameDel << std::string(padding_length, kPadding) << ' ';

  std::string alloc_size = to_megabytes(node.alloc_size);
  out << alloc_size << std::string(stats.max_alloc - alloc_size.length(), ' ') << ' ' << node.allocs
      << std::string(stats.max_allocs - std::to_string(node.allocs).length(), ' ') << ' '
      << node.frees << std::string(stats.max_frees - std::to_string(node.frees).length(), ' ')
      << '\n';

  if (!node.ordered_children.empty()) {
    auto last_child = node.ordered_children.back();

    for (auto const &child : node.ordered_children) {
      const bool is_last = (child == last_child);
      print_heap_tree_node(out, *child, stats, depth + 1, is_last);
    }
  }
}
} // namespace kaminpar::heap_profiler