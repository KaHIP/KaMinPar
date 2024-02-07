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

namespace kaminpar::heap_profiler {

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
    std::string_view var_name, std::string_view file_name, std::size_t line
) {
  if (_enabled) {
    _var_name = var_name;
    _file_name = file_name;
    _line = line;
  }
}

DataStructure *HeapProfiler::add_data_struct(std::string name, std::size_t size) {
  if (_enabled) {
    std::lock_guard<std::mutex> guard(_mutex);

    DataStructure *data_structure = _struct_allocator.create(std::move(name), size);
    if (_line != 0) {
      data_structure->variable_name = _var_name;
      data_structure->file_name = _file_name;
      data_structure->line = _line;

      _line = 0;
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
      node->allocs++;
      node->alloc_size += size;

      if (std::size_t current_alloc = node->alloc_size - node->free_size;
          node->alloc_size > node->free_size && current_alloc > node->max_alloc_size) {
        node->max_alloc_size = current_alloc;
      }
    }

    _address_map.insert_or_assign(ptr, size);
  }
}

void HeapProfiler::record_free(const void *ptr) {
  if (_enabled) {
    std::lock_guard<std::mutex> guard(_mutex);

    if (auto search = _address_map.find(ptr); search != _address_map.end()) {
      std::size_t size = search->second;
      for (HeapProfileTreeNode *node = _tree.currentNode; node != nullptr; node = node->parent) {
        node->frees++;
        node->free_size += size;
      }

      _address_map.erase(search);
    }
  }
}

void HeapProfiler::set_detailed_summary_options() {
  set_max_depth(std::numeric_limits<std::size_t>::max());
  set_print_data_structs(true);
  set_min_data_struct_size(0);
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

  stats.max_alloc_size =
      std::max(kMaxAllocTitle.length(), to_megabytes(stats.max_alloc_size).length());
  stats.alloc_size = std::max(kAllocTitle.length(), to_megabytes(stats.alloc_size).length());
  stats.free_size = std::max(kAllocTitle.length(), to_megabytes(stats.free_size).length());
  stats.allocs = std::max(kAllocsTitle.length(), std::to_string(stats.allocs).length());
  stats.frees = std::max(kFreesTitle.length(), std::to_string(stats.frees).length());

  out << std::string(stats.len + kNameDel.length() + kPercentageLength - 1, kHeadingPadding) << ' ';
  out << kMaxAllocTitle << std::string(stats.max_alloc_size - kMaxAllocTitle.length() + 1, ' ');
  out << kAllocTitle << std::string(stats.alloc_size - kAllocTitle.length() + 1, ' ');
  out << kFreeTitle << std::string(stats.free_size - kFreeTitle.length() + 1, ' ');
  out << kAllocsTitle << std::string(stats.allocs - kAllocsTitle.length() + 1, ' ');
  out << kFreesTitle << std::string(stats.frees - kFreesTitle.length() + 1, ' ');
  out << '\n';

  print_heap_tree_node(out, root, stats, _max_depth, _print_data_structs, _min_data_struct_size);
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
  print_percentage(out, node);

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
    std::size_t leading_whitespaces = (depth - 1) * kBranchLength;
    out << std::string(leading_whitespaces, ' ') << (last ? kTailBranch : kBranch);
  }
}

void HeapProfiler::print_percentage(std::ostream &out, const HeapProfileTreeNode &node) {
  std::size_t parent_alloc_size = node.parent == nullptr ? 0 : node.parent->alloc_size;
  float percentage = (parent_alloc_size == 0) ? 1 : (node.alloc_size / (float)parent_alloc_size);

  out << "(";

  if (percentage >= 0.999995) {
    out << "100.00";
  } else {
    if (percentage < 0.1) {
      out << "0";
    }

    out << percentage * 100;
  }

  out << "%) ";
}

void HeapProfiler::print_statistics(
    std::ostream &out, const HeapProfileTreeNode &node, const HeapProfileTreeStats stats
) {
  std::string max_alloc_size = to_megabytes(node.max_alloc_size);
  out << max_alloc_size << std::string(stats.max_alloc_size - max_alloc_size.length() + 1, ' ');

  std::string alloc_size = to_megabytes(node.alloc_size);
  out << alloc_size << std::string(stats.alloc_size - alloc_size.length() + 1, ' ');

  std::string free_size = to_megabytes(node.free_size);
  out << free_size << std::string(stats.free_size - free_size.length() + 1, ' ');

  out << node.allocs << std::string(stats.allocs - std::to_string(node.allocs).length() + 1, ' ')
      << node.frees << std::string(stats.frees - std::to_string(node.frees).length(), ' ') << '\n';
}

void HeapProfiler::print_data_structures(
    std::ostream &out,
    const HeapProfileTreeNode &node,
    std::size_t depth,
    bool last,
    std::size_t min_data_struct_size
) {
  std::vector<DataStructure *, NoProfilAllocator<DataStructure *>> filtered_data_structures;
  std::copy_if(
      node.data_structures.begin(),
      node.data_structures.end(),
      std::back_inserter(filtered_data_structures),
      [&](auto *data_structure) { return data_structure->size >= min_data_struct_size; }
  );

  std::sort(
      filtered_data_structures.begin(),
      filtered_data_structures.end(),
      [](auto *d1, auto *d2) { return d1->size > d2->size; }
  );

  auto last_data_structure = filtered_data_structures.back();
  for (auto data_structure : filtered_data_structures) {
    const bool is_last = last && (data_structure == last_data_structure);
    const bool has_info = data_structure->line > 0;

    std::size_t leading_whitespaces = depth * kBranchLength;
    out << std::string(leading_whitespaces, ' ') << (is_last ? kTailBranch : kBranch);

    std::size_t max_alloc_size = node.max_alloc_size;
    float percentage = (max_alloc_size == 0) ? 1 : (data_structure->size / (float)max_alloc_size);
    if (percentage <= 1) {
      out << '(' << (percentage * 100) << "%) ";
    }

    out << data_structure->name;
    if (has_info) {
      out << " \"" << data_structure->variable_name << '\"';
    }
    out << " uses " << to_megabytes(data_structure->size) << " mb ";

    if (has_info) {
      out << " (" << data_structure->file_name << " at line " << data_structure->line << ')';
    }

    out << '\n';
  }
}

} // namespace kaminpar::heap_profiler
