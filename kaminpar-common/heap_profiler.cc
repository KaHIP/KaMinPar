/*******************************************************************************
 * Heap profiler to measure heap memory usage.
 *
 * @file:   heap_profiler.cc
 * @author: Daniel Salwasser
 * @date:   21.10.2023
 ******************************************************************************/
#include "kaminpar-common/heap_profiler.h"

#include <kassert/kassert.hpp>

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
  static HeapProfiler global("Global Heap Profiler");
  return global;
}

HeapProfiler::HeapProfiler(std::string_view name) {
  _tree.root.name = name;
  _tree.root.parent = nullptr;
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
  HeapProfileTreeNode *node = _node_allocator.construct();
  node->name = name;
  node->description = description;
  node->parent = _tree.currentNode;

  _tree.currentNode->children.push_back(node);
  _tree.currentNode = node;
}

void HeapProfiler::stop_profile() {
  KASSERT(_tree.currentNode->parent != nullptr, "The root heap profile cannot be stopped.");
  _tree.currentNode = _tree.currentNode->parent;
}

ScopedHeapProfiler
HeapProfiler::start_scoped_profile(std::string_view name, std::string description) {
  return ScopedHeapProfiler(name, description);
}

void HeapProfiler::record_alloc(const void *ptr, std::size_t size) {
  if (_enabled) {
    std::lock_guard<std::mutex> guard(_mutex);

    for (HeapProfileTreeNode *node = _tree.currentNode; node != nullptr; node = node->parent) {
      node->allocs++;
      node->alloc_size += size;

      if (std::size_t current_alloc = node->alloc_size - node->free_size;
          current_alloc > node->max_alloc_size) {
        node->max_alloc_size = current_alloc;
      }
    }

    _address_map[ptr] = size;
  }
}

void HeapProfiler::record_free(const void *ptr) {
  if (_enabled) {
    std::lock_guard<std::mutex> guard(_mutex);

    std::size_t size = _address_map[ptr];
    for (HeapProfileTreeNode *node = _tree.currentNode; node != nullptr; node = node->parent) {
      node->frees++;
      node->free_size += size;
    }

    _address_map.erase(ptr);
  }
}

void HeapProfiler::print_heap_profile(std::ostream &out) {
  HeapProfileTreeNode &root = *_tree.currentNode;
  HeapProfileTreeStats stats(root);

  stats.max_alloc_size =
      std::max(kMaxAllocTitle.length(), to_megabytes(stats.max_alloc_size).length());
  stats.alloc_size = std::max(kAllocTitle.length(), to_megabytes(stats.alloc_size).length());
  stats.free_size = std::max(kAllocTitle.length(), to_megabytes(stats.free_size).length());
  stats.allocs = std::max(kAllocsTitle.length(), std::to_string(stats.allocs).length());
  stats.frees = std::max(kFreesTitle.length(), std::to_string(stats.frees).length());

  out << std::string(stats.len + 2 + 10, '-') << ' ';
  out << kMaxAllocTitle << std::string(stats.max_alloc_size - kMaxAllocTitle.length() + 1, ' ');
  out << kAllocTitle << std::string(stats.alloc_size - kAllocTitle.length() + 1, ' ');
  out << kFreeTitle << std::string(stats.free_size - kFreeTitle.length() + 1, ' ');
  out << kAllocsTitle << std::string(stats.allocs - kAllocsTitle.length() + 1, ' ');
  out << kFreesTitle << std::string(stats.frees - kFreesTitle.length() + 1, ' ');
  out << '\n';

  print_heap_tree_node(out, root, stats);
  out << '\n';
}

std::size_t HeapProfiler::get_max_alloc() {
  return _tree.currentNode->max_alloc_size;
}

std::size_t HeapProfiler::get_alloc() {
  return _tree.currentNode->alloc_size;
}

std::size_t HeapProfiler::get_free() {
  return _tree.currentNode->free_size;
}

std::size_t HeapProfiler::get_allocs() {
  return _tree.currentNode->allocs;
}

std::size_t HeapProfiler::get_frees() {
  return _tree.currentNode->frees;
}

void HeapProfiler::print_heap_tree_node(
    std::ostream &out,
    const HeapProfileTreeNode &node,
    const HeapProfileTreeStats stats,
    std::size_t depth,
    bool last
) {
  float percentage;
  if (depth > 0) {
    std::size_t leading_whitespaces = (depth - 1) * kBranchLength;
    out << std::string(leading_whitespaces, ' ') << (last ? kTailBranch : kBranch);

    std::size_t parent_alloc_size = node.parent->alloc_size;
    percentage = (parent_alloc_size == 0) ? 1 : (node.alloc_size / (float)parent_alloc_size);
  } else {
    percentage = 1;
  }

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

  std::size_t padding_length = stats.len - (depth * kBranchLength + node.name.length());
  if (!node.description.empty()) {
    padding_length -= node.description.length() + 2;
    out << "(" << node.description << ")";
  }

  out << kNameDel << std::string(padding_length, kPadding) << ' ';

  std::string max_alloc_size = to_megabytes(node.max_alloc_size);
  out << max_alloc_size << std::string(stats.max_alloc_size - max_alloc_size.length() + 1, ' ');

  std::string alloc_size = to_megabytes(node.alloc_size);
  out << alloc_size << std::string(stats.alloc_size - alloc_size.length() + 1, ' ');

  std::string free_size = to_megabytes(node.free_size);
  out << free_size << std::string(stats.free_size - free_size.length() + 1, ' ');

  out << node.allocs << std::string(stats.allocs - std::to_string(node.allocs).length() + 1, ' ')
      << node.frees << std::string(stats.frees - std::to_string(node.frees).length(), ' ') << '\n';

  if (!node.children.empty()) {
    auto last_child = node.children.back();

    for (auto const &child : node.children) {
      const bool is_last = (child == last_child);
      print_heap_tree_node(out, *child, stats, depth + 1, is_last);
    }
  }
}
} // namespace kaminpar::heap_profiler
