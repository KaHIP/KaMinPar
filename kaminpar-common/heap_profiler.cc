/*******************************************************************************
 * Heap profiler to measure heap memory usage.
 *
 * @file:   heap_profiler.cc
 * @author: Daniel Salwasser
 * @date:   21.10.2023
 ******************************************************************************/
#include "kaminpar-common/heap_profiler.h"

#include <algorithm>

#include <kassert/kassert.hpp>

#ifdef __linux__
#include <unistd.h>
#elif _WIN32
#include <windows.h>
#else
#include <limits>
#endif

namespace kaminpar::heap_profiler {

// Source: https://stackoverflow.com/a/2513561
#ifdef __linux__
std::size_t get_total_system_memory() {
  const long npages = sysconf(_SC_PHYS_PAGES);
  const long pagesz = sysconf(_SC_PAGE_SIZE);

  if (npages <= 0 || pagesz <= 0) {
    return std::numeric_limits<std::size_t>::max();
  }

  return static_cast<std::size_t>(npages * pagesz);
}
#elif _WIN32
std::size_t get_total_system_memory() {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return static_cast<std::size_t>(status.ullTotalPhys);
}
#else
std::size_t get_total_system_memory() {
  return std::numeric_limits<std::size_t>::max();
}
#endif

HeapProfiler &HeapProfiler::global() {
  static HeapProfiler global("Global Heap Profiler");
  return global;
}

HeapProfiler::HeapProfiler(std::string_view name) : _tree(name) {}

HeapProfiler::~HeapProfiler() {
  _tree.root.free(_node_allocator, _struct_allocator);
}

void HeapProfiler::enable() {
  _enabled = true;
}

void HeapProfiler::disable() {
  _enabled = false;
}

void HeapProfiler::start_profile(std::string_view name, std::string desc) {
  if (_enabled) {
    std::lock_guard<std::mutex> guard(_mutex);

    HeapProfileTreeNode *node = _node_allocator.create(name, desc, _tree.currentNode);
    _tree.currentNode->children.push_back(node);
    _tree.currentNode = node;
  }
}

void HeapProfiler::stop_profile() {
  if (_enabled) {
    KASSERT(_tree.currentNode->parent != nullptr, "The root heap profile cannot be stopped.");
    std::lock_guard<std::mutex> guard(_mutex);

    _tree.currentNode = _tree.currentNode->parent;
  }
}

ScopedHeapProfiler HeapProfiler::start_scoped_profile(std::string_view name, std::string desc) {
  return ScopedHeapProfiler(name, desc);
}

void HeapProfiler::record_data_struct(
    std::string_view var_name, const std::source_location location
) {
  if (_enabled) {
    _var_name = var_name;
    _file_name = location.file_name();
    _line = location.line();
    _column = location.column();
  }
}

DataStructure *HeapProfiler::add_data_struct(hp_string name, std::size_t size) {
  if (_enabled) {
    std::lock_guard<std::mutex> guard(_mutex);

    DataStructure *data_structure = _struct_allocator.create(std::move(name), size);
    if (_line != 0) {
      data_structure->variable_name = _var_name;
      data_structure->file_name = _file_name;
      data_structure->line = _line;
      data_structure->column = _column;

      _line = 0;
      _column = 0;
    }

    _tree.currentNode->data_structures.push_back(data_structure);
    return data_structure;
  }

  return new DataStructure(std::move(name), size);
}

void HeapProfiler::record_alloc(const void *ptr, std::size_t size) {
  if (_enabled) {
    std::lock_guard<std::mutex> guard(_mutex);

    for (HeapProfileTreeNode *node = _tree.currentNode; node != nullptr; node = node->parent) {
      node->num_allocs++;
      node->total_alloc += size;

      if (std::size_t current_alloc = node->total_alloc - node->total_free;
          node->total_alloc > node->total_free && current_alloc > node->peak_memory) {
        node->peak_memory = current_alloc;
      }
    }

    _address_map.insert_or_assign(ptr, size);
  }
}

void HeapProfiler::record_free(const void *ptr) {
  if (_enabled) {
    std::lock_guard<std::mutex> guard(_mutex);

    if (auto search = _address_map.find(ptr); search != _address_map.end()) {
      const std::size_t size = search->second;

      for (HeapProfileTreeNode *node = _tree.currentNode; node != nullptr; node = node->parent) {
        node->num_frees++;
        node->total_free += size;
      }

      _address_map.erase(search);
    }
  }
}

void HeapProfiler::set_experiment_summary_options() {
  set_max_depth(std::numeric_limits<std::size_t>::max());
  set_print_data_structs(false);
}

void HeapProfiler::set_max_depth(std::size_t max_depth) {
  _max_depth = max_depth;
}

void HeapProfiler::set_print_data_structs(bool print) {
  _print_data_structs = print;
}

void HeapProfiler::set_min_data_struct_size(float size) {
  _min_data_struct_size = static_cast<std::size_t>(size * 1024 * 1024);
}

void HeapProfiler::print_heap_profile(std::ostream &out) {
  HeapProfileTreeNode &root = *_tree.currentNode;
  HeapProfileTreeStats stats(root);

  stats.peak_memory = std::max(kMaxAllocTitle.length(), to_megabytes(stats.peak_memory).length());
  stats.total_alloc = std::max(kAllocTitle.length(), to_megabytes(stats.total_alloc).length());
  stats.total_free = std::max(kAllocTitle.length(), to_megabytes(stats.total_free).length());
  stats.num_allocs = std::max(kAllocsTitle.length(), std::to_string(stats.num_allocs).length());
  stats.num_frees = std::max(kFreesTitle.length(), std::to_string(stats.num_frees).length());

  out << std::string(stats.len + kNameDel.length() + kPercentageLength - 1, kHeadingPadding) << ' ';
  out << kMaxAllocTitle << std::string(stats.peak_memory - kMaxAllocTitle.length() + 1, ' ');
  out << kAllocTitle << std::string(stats.total_alloc - kAllocTitle.length() + 1, ' ');
  out << kFreeTitle << std::string(stats.total_free - kFreeTitle.length() + 1, ' ');
  out << kAllocsTitle << std::string(stats.num_allocs - kAllocsTitle.length() + 1, ' ');
  out << kFreesTitle << std::string(stats.num_frees - kFreesTitle.length() + 1, ' ');
  if (!_tree.annotation.empty()) {
    out << "   " << _tree.annotation;
  }
  out << '\n';

  print_heap_tree_node(out, root, stats, _max_depth, _print_data_structs, _min_data_struct_size);
  out << '\n';
}

std::size_t HeapProfiler::peak_memory() {
  return _tree.currentNode->peak_memory;
}

std::size_t HeapProfiler::total_alloc() {
  return _tree.currentNode->total_alloc;
}

std::size_t HeapProfiler::total_free() {
  return _tree.currentNode->total_free;
}

std::size_t HeapProfiler::num_allocs() {
  return _tree.currentNode->num_allocs;
}

std::size_t HeapProfiler::num_frees() {
  return _tree.currentNode->num_frees;
}

[[nodiscard]] HeapProfiler::HeapProfileTree &HeapProfiler::tree_root() {
  return _tree;
}

void HeapProfiler::print_heap_tree_node(
    std::ostream &out,
    const HeapProfileTreeNode &node,
    const HeapProfileTreeStats stats,
    std::size_t max_depth,
    bool print_data_structs,
    std::size_t min_data_struct_size,
    std::size_t depth,
    bool last
) {
  if (depth > max_depth) {
    return;
  }

  print_indentation(out, depth, last);

  const std::size_t parent_alloc_size = node.parent == nullptr ? 0 : node.parent->total_alloc;
  const float percentage =
      (parent_alloc_size == 0) ? 1 : (node.total_alloc / static_cast<float>(parent_alloc_size));
  print_percentage(out, percentage);

  out << node.name;

  std::size_t padding_length = stats.len - (depth * kBranchLength + node.name.length());
  if (!node.description.empty()) {
    padding_length -= node.description.length() + 2;
    out << '(' << node.description << ')';
  }

  out << kNameDel;
  if (padding_length > 0) {
    out << std::string(padding_length - 1, kPadding) << ' ';
  }

  print_statistics(out, node, stats);
  if (print_data_structs) {
    print_data_structures(out, node, depth, node.children.empty(), min_data_struct_size);
  }

  if (!node.children.empty()) {
    const auto last_child = node.children.back();

    for (auto const &child : node.children) {
      const bool is_last = (child == last_child);
      print_heap_tree_node(
          out,
          *child,
          stats,
          max_depth,
          print_data_structs,
          min_data_struct_size,
          depth + 1,
          is_last
      );
    }
  }
}

void HeapProfiler::print_indentation(std::ostream &out, std::size_t depth, bool last) {
  if (depth > 0) {
    const std::size_t leading_whitespaces = (depth - 1) * kBranchLength;
    out << std::string(leading_whitespaces, ' ') << (last ? kTailBranch : kBranch);
  }
}

void HeapProfiler::print_percentage(std::ostream &out, const float percentage) {
  out << '(';

  if (percentage >= 0.99995) {
    out << "100.0";
  } else {
    if (percentage < 0.1) {
      out << '0';
    }

    out << std::fixed << std::setprecision(2) << percentage * 100;
  }

  out << "%) ";
}

void HeapProfiler::print_statistics(
    std::ostream &out, const HeapProfileTreeNode &node, const HeapProfileTreeStats stats
) {
  const std::string peak_memory_size = to_megabytes(node.peak_memory);
  out << peak_memory_size << std::string(stats.peak_memory - peak_memory_size.length() + 1, ' ');

  const std::string total_alloc_size = to_megabytes(node.total_alloc);
  out << total_alloc_size << std::string(stats.total_alloc - total_alloc_size.length() + 1, ' ');

  const std::string total_free_size = to_megabytes(node.total_free);
  out << total_free_size << std::string(stats.total_free - total_free_size.length() + 1, ' ');

  out << node.num_allocs
      << std::string(stats.num_allocs - std::to_string(node.num_allocs).length() + 1, ' ')
      << node.num_frees
      << std::string(stats.num_frees - std::to_string(node.num_frees).length(), ' ');

  if (!node.annotation.empty()) {
    out << "   " << node.annotation;
  }

  out << '\n';
}

void HeapProfiler::print_data_structures(
    std::ostream &out,
    const HeapProfileTreeNode &node,
    std::size_t depth,
    bool last,
    std::size_t min_data_struct_size
) {
  std::vector<DataStructure *, NoProfileAllocator<DataStructure *>> filtered_data_structures;
  std::copy_if(
      node.data_structures.begin(),
      node.data_structures.end(),
      std::back_inserter(filtered_data_structures),
      [&](auto *data_structure) { return data_structure->size >= min_data_struct_size; }
  );

  if (filtered_data_structures.empty()) {
    return;
  }

  std::sort(
      filtered_data_structures.begin(),
      filtered_data_structures.end(),
      [](auto *d1, auto *d2) { return d1->size > d2->size; }
  );

  auto last_data_structure = filtered_data_structures.back();
  for (auto data_structure : filtered_data_structures) {
    const bool is_last = last && (data_structure == last_data_structure);
    const bool has_info = data_structure->line > 0;

    const std::size_t leading_whitespaces = depth * kBranchLength;
    out << std::string(leading_whitespaces, ' ') << (is_last ? kTailBranch : kBranch);

    const float percentage =
        (node.peak_memory == 0) ? 1 : (data_structure->size / static_cast<float>(node.peak_memory));
    print_percentage(out, percentage);

    out << data_structure->name;
    if (has_info) {
      out << " \"" << data_structure->variable_name << '\"';
    }
    out << " uses " << to_megabytes(data_structure->size) << " mb ";

    if (has_info) {
      out << " [" << data_structure->file_name << '(' << data_structure->line << ':'
          << data_structure->column << ")]";
    }

    out << '\n';
  }
}

} // namespace kaminpar::heap_profiler
