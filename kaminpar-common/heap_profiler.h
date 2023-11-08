/*******************************************************************************
 * Heap profiler to measure heap memory usage.
 *
 * @file:   heap_profiler.h
 * @author: Daniel Salwasser
 * @date:   21.10.2023
 ******************************************************************************/
#pragma once

#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifdef KAMINPAR_ENABLE_HEAP_PROFILING

#include "kaminpar-common/libc_memory_override.h"

// A macro to get the path of a source file in the project directory
// (https://stackoverflow.com/a/40947954)
#define __FILENAME__ (__FILE__ + SOURCE_PATH_SIZE)

#define GET_MACRO(X, Y, Z, FUNC, ...) FUNC
#define START_HEAP_PROFILER_2(name, desc)                                                          \
  kaminpar::heap_profiler::HeapProfiler::global().start_profile(name, desc)
#define START_HEAP_PROFILER_1(name) START_HEAP_PROFILER_2(name, "")
#define START_HEAP_PROFILER(...)                                                                   \
  GET_MACRO(_, ##__VA_ARGS__, START_HEAP_PROFILER_2, START_HEAP_PROFILER_1)(__VA_ARGS__)
#define STOP_HEAP_PROFILER() kaminpar::heap_profiler::HeapProfiler::global().stop_profile()
#define SCOPED_HEAP_PROFILER_2(name, desc, line)                                                   \
  auto __SCOPED_HEAP_PROFILER__##line =                                                            \
      kaminpar::heap_profiler::HeapProfiler::global().start_scoped_profile(name, desc)
#define SCOPED_HEAP_PROFILER_1(name, line) SCOPED_HEAP_PROFILER_2(name, "", line)
#define SCOPED_HEAP_PROFILER(...)                                                                  \
  GET_MACRO(_, ##__VA_ARGS__, SCOPED_HEAP_PROFILER_2, SCOPED_HEAP_PROFILER_1)(__VA_ARGS__, __LINE__)
#define RECORD_DATA_STRUCT_2(name, size, variable_name)                                            \
  variable_name = kaminpar::heap_profiler::HeapProfiler::global().add_data_struct(name, size)
#define RECORD_DATA_STRUCT_1(name, size)                                                           \
  kaminpar::heap_profiler::HeapProfiler::global().add_data_struct(name, size)
#define RECORD_DATA_STRUCT(...)                                                                    \
  GET_MACRO(__VA_ARGS__, RECORD_DATA_STRUCT_2, RECORD_DATA_STRUCT_1)(__VA_ARGS__)
#define RECORD(name)                                                                               \
  kaminpar::heap_profiler::HeapProfiler::global().record_data_struct(name, __FILENAME__, __LINE__);
#define IF_HEAP_PROFILING(expression) expression
#define ENABLE_HEAP_PROFILER() kaminpar::heap_profiler::HeapProfiler::global().enable()
#define DISABLE_HEAP_PROFILER() kaminpar::heap_profiler::HeapProfiler::global().disable()
#define PRINT_HEAP_PROFILE(out)                                                                    \
  kaminpar::heap_profiler::HeapProfiler::global().print_heap_profile(out)

/*!
 * Whether heap profiling is enabled.
 */
constexpr bool kHeapProfiling = true;

#else

#define START_HEAP_PROFILER(...)
#define STOP_HEAP_PROFILER()
#define SCOPED_HEAP_PROFILER(...)
#define RECORD_DATA_STRUCT(...)
#define RECORD(...)
#define IF_HEAP_PROFILING(...)
#define ENABLE_HEAP_PROFILER()
#define DISABLE_HEAP_PROFILER()
#define PRINT_HEAP_PROFILE(...)

/*!
 * Whether heap profiling is enabled.
 */
constexpr bool kHeapProfiling = false;

#endif

namespace kaminpar::heap_profiler {

/*!
 * A minimal allocator that uses memory allocation functions which bypass the heap profiler.
 *
 * This is required for allocations inside the heap profiler, otherwise a memory allocation would
 * lead to an infinite recursion.
 */
template <typename T> struct NoProfilAllocator {
  using value_type = T;

  NoProfilAllocator() noexcept {}
  template <typename U> NoProfilAllocator(const NoProfilAllocator<U> &) noexcept {}

  template <typename U> bool operator==(const NoProfilAllocator<U> &) const noexcept {
    return true;
  }
  template <typename U> bool operator!=(const NoProfilAllocator<U> &) const noexcept {
    return false;
  }

  T *allocate(const size_t n) const {
    if (n == 0) {
      return nullptr;
    }

    if (n > static_cast<size_t>(-1) / sizeof(T)) {
      throw std::bad_array_new_length();
    }

#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
    void *const ptr = std_malloc(n * sizeof(T));
#else
    void *const ptr = std::malloc(n * sizeof(T));
#endif
    if (!ptr) {
      throw std::bad_alloc();
    }

    return static_cast<T *>(ptr);
  }

  void deallocate(T *const ptr, size_t) const noexcept {
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
    std_free(ptr);
#else
    std::free(ptr);
#endif
  }

  template <typename... Args> T *create(Args &&...args) const {
    T *t = allocate(1);
    new (t) T(std::forward<Args>(args)...);
    return t;
  }

  void destruct(T *const t) const {
    t->~T();
    deallocate(t, 1);
  }
};

/*!
 * Represents a data structure in the program. It contains information about a data structure that
 * is tracked by the heap profiler.
 */
struct DataStructure {
  /*!
   * The name of the data structure.
   */
  std::string_view name;
  /*!
   * The size of the memory in bytes allocated on the heap by the data structure.
   */
  std::size_t size;

  /*!
   * The name of the variable storing the data structure. It is empty if it is not available.
   */
  std::string_view variable_name;
  /*!
   * The name of the source file of the variable storing the data structure. It is empty if it is
   * not available.
   */
  std::string_view file_name;
  /*!
   * The line of the variable storing the data structure. It is zero if it is not available.
   */
  std::size_t line;

  /*!
   * Constructs a new data structure.
   *
   * @param name The name of the data structure.
   * @param size The size of the memory in bytes allocated on the heap by the data structure.
   */
  explicit DataStructure(std::string_view name, std::size_t size)
      : name(name),
        size(size),
        variable_name(""),
        file_name(""),
        line(0) {}
};

class ScopedHeapProfiler;

/*!
 * A hierarchical heap profiler to measure dynamic memory allocation of the program.
 *
 * The memory allocation operations of libc are overridden to additionally call the global heap
 * profiler on each allocation and deallocation request.
 */
class HeapProfiler {
private:
  static constexpr std::string_view kMaxAllocTitle = "Max Alloc (mb)";
  static constexpr std::string_view kAllocTitle = "Alloc (mb)";
  static constexpr std::string_view kFreeTitle = "Free (mb)";
  static constexpr std::string_view kAllocsTitle = "Allocs";
  static constexpr std::string_view kFreesTitle = "Frees";

  static constexpr std::string_view kBranch = "|-- ";
  static constexpr std::string_view kTailBranch = "`-- ";
  static constexpr std::string_view kTailEdge = "    ";
  static constexpr std::string_view kNameDel = ": ";
  static constexpr char kHeadingPadding = '-';
  static constexpr char kPadding = '.';

  static constexpr std::size_t kBranchLength = 4;
  static constexpr std::size_t kPercentageLength = 10;
  static constexpr std::size_t kDataStructSizeThreshold = 1024;

  static std::string to_megabytes(std::size_t bytes) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << (bytes / (float)(1024 * 1024));
    return stream.str();
  }

  struct HeapProfileTreeNode {
    std::string_view name;
    std::string description;

    HeapProfileTreeNode *parent;
    std::vector<HeapProfileTreeNode *, NoProfilAllocator<HeapProfileTreeNode *>> children;

    std::size_t max_alloc_size;
    std::size_t alloc_size;
    std::size_t free_size;
    std::size_t allocs;
    std::size_t frees;

    std::vector<DataStructure *, NoProfilAllocator<DataStructure *>> data_structures;

    HeapProfileTreeNode(std::string_view name, std::string description, HeapProfileTreeNode *parent)
        : name(name),
          description(description),
          parent(parent),
          max_alloc_size(0),
          alloc_size(0),
          free_size(0),
          allocs(0),
          frees(0) {}

    template <typename NodeAllocator, typename DataStructAllocator>
    void free(NodeAllocator node_allocator, DataStructAllocator data_struct_allocator) {
      for (DataStructure *data_structure : data_structures) {
        data_struct_allocator.destruct(data_structure);
      }

      for (HeapProfileTreeNode *child : children) {
        child->free(node_allocator, data_struct_allocator);
        node_allocator.destruct(child);
      }
    }
  };

  struct HeapProfileTree {
    HeapProfileTreeNode root;
    HeapProfileTreeNode *currentNode;

    HeapProfileTree(std::string_view name) : root(name, "", nullptr), currentNode(&root) {}
  };

  struct HeapProfileTreeStats {
    std::size_t len;
    std::size_t max_alloc_size;
    std::size_t alloc_size;
    std::size_t free_size;
    std::size_t allocs;
    std::size_t frees;

    HeapProfileTreeStats(const HeapProfileTreeNode &node) {
      std::size_t name_length = node.name.length();
      if (!node.description.empty()) {
        name_length += node.description.length() + 2;
      }

      len = name_length;
      max_alloc_size = node.max_alloc_size;
      alloc_size = node.alloc_size;
      free_size = node.free_size;
      allocs = node.allocs;
      frees = node.frees;

      for (auto const &child : node.children) {
        HeapProfileTreeStats child_stats(*child);

        len = std::max(len, child_stats.len + kBranchLength);
        max_alloc_size = std::max(max_alloc_size, child_stats.max_alloc_size);
        alloc_size = std::max(alloc_size, child_stats.alloc_size);
        free_size = std::max(free_size, child_stats.free_size);
        allocs = std::max(allocs, child_stats.allocs);
        frees = std::max(frees, child_stats.frees);
      }
    }
  };

public:
  /**
   * Returns the global heap profiler.
   *
   * @return The global heap profiler.
   */
  static HeapProfiler &global();

  /*!
   * Constructs a new heap profiler.
   *
   * @param name The name of the heap profiler and the name of the root profile.
   */
  explicit HeapProfiler(std::string_view name);

  /*!
   * Destroys the heap profiler.
   */
  ~HeapProfiler();

  /*!
   * Starts profiling the heap.
   */
  void enable();

  /*!
   * Stops profiling the heap.
   */
  void disable();

  /**
   * Starts a new profile, adds it as a child profile to the current profile, and sets it to the
   * current profile.
   *
   * @param name The name of the profile to start.
   * @param desc The description of the profile to start.
   */
  void start_profile(std::string_view name, std::string desc);

  /*!
   * Stops the current profile and sets the new current profile to the parent profile.
   */
  void stop_profile();

  /*!
   * Starts a scoped heap profile and returns the associated object.
   *
   * @param name The name of the profile to start.
   * @param desc The description of the profile to start.
   */
  ScopedHeapProfiler start_scoped_profile(std::string_view name, std::string desc);

  /*!
   * Records information about the variable storing the next data structure that is added to the
   * heap profiler.
   *
   * @param var_name The name of the variable storing the data structure.
   * @param file_name The name of the source file of the variable storing the data structure.
   * @param line The line of the variable storing the data structure.
   */
  void record_data_struct(std::string_view var_name, std::string_view file_name, std::size_t line);

  /*!
   * Adds a data structure to track to the current profile of the heap profiler. If information
   * about the variable that stores the data structure has been recorded by the heap profiler, it is
   * added.
   *
   * @param name The name of the data structure.
   * @param size The size of the memory in bytes allocated on the heap by the data structure.
   * @return A pointer to the object holding information about the data structure or a nullptr if
   * the heap profiler is disabled.
   */
  DataStructure *add_data_struct(std::string_view name, std::size_t size);

  /*!
   * Records a memory allocation.
   *
   * @param ptr The pointer to the beginning of the allocated memory.
   * @param size The number allocated bytes.
   */
  void record_alloc(const void *ptr, std::size_t size);

  /*!
   * Records a memory deallocation.
   *
   * @param ptr The pointer to the beginning of the allocated memory
   */
  void record_free(const void *ptr);

  /*!
   * Sets the maximum depth shown in the summary.
   *
   * @param max_depth The maximum depth shown in the summary.
   */
  void set_max_depth(std::size_t max_depth);

  /*!
   * Sets the option whether to print data structure memory statistics in the summary.
   *
   * @param print Whether to print data structure memory statistics in the summary.
   */
  void set_print_data_structs(bool print);

  /*!
   * Sets the option whether to print all data structure memory statistics in the summary.
   *
   * @param print Whether to print all data structure memory statistics in the summary.
   */
  void set_print_all_data_structs(bool print);

  /*!
   * Prints information about the heap profile to the output stream.
   *
   * @param out The output stream to write to.
   */
  void print_heap_profile(std::ostream &out);

  /*!
   * Returns the amount of maximum allocated memory in bytes of the current heap profile.
   *
   * @return The amount of maximum allocated memory in bytes of the current heap profile.
   */
  std::size_t get_max_alloc();

  /*!
   * Returns the amount of allocated memory in bytes of the current heap profile.
   *
   * @return The amount of allocated memory in bytes of the current heap profile.
   */
  std::size_t get_alloc();

  /*!
   * Returns the amount of freed memory in bytes of the current heap profile.
   *
   * @return The amount of freed memory in bytes of the current heap profile.
   */
  std::size_t get_free();

  /*!
   * Returns the amount of alloc operations of the current heap profile.
   *
   * @return The amount of alloc operations of the current heap profile.
   */
  std::size_t get_allocs();

  /*!
   * Returns the amount of free operations of the current heap profile.
   *
   * @return The amount of free operations of the current heap profile.
   */
  std::size_t get_frees();

private:
  bool _enabled = false;
  std::mutex _mutex;

  NoProfilAllocator<HeapProfileTreeNode> _node_allocator;
  HeapProfileTree _tree;
  std::unordered_map<
      const void *,
      std::size_t,
      std::hash<const void *>,
      std::equal_to<const void *>,
      NoProfilAllocator<std::pair<const void *const, std::size_t>>>
      _address_map;

  NoProfilAllocator<DataStructure> _struct_allocator;
  std::string_view _var_name;
  std::string_view _file_name;
  std::size_t _line;

  std::size_t _max_depth = std::numeric_limits<std::size_t>::max();
  bool _print_data_structs = true;
  bool _print_all_data_structs = true;

  static void print_heap_tree_node(
      std::ostream &out,
      const HeapProfileTreeNode &node,
      const HeapProfileTreeStats stats,
      std::size_t max_depth,
      bool print_data_structs,
      bool print_all_data_structs,
      std::size_t depth = 0,
      bool last = false
  );
  static void print_indentation(std::ostream &out, std::size_t depth, bool last);
  static void print_percentage(std::ostream &out, const HeapProfileTreeNode &node);
  static void print_statistics(
      std::ostream &out, const HeapProfileTreeNode &node, const HeapProfileTreeStats stats
  );
  static void print_data_structures(
      std::ostream &out,
      const HeapProfileTreeNode &node,
      std::size_t depth,
      bool last,
      bool print_all_data_structs
  );
};

/*!
 * A helper class for scoped heap profiling. The profile starts with the construction of the object
 * and ends with the destruction of the object.
 */
class ScopedHeapProfiler {
public:
  /*!
   * Constructs a new scoped heap profiler and thereby starting a new heap profile.
   *
   * @param name The name of the started profile.
   * @param description The description of the started profile.
   */
  ScopedHeapProfiler(std::string_view name, std::string description) {
    HeapProfiler::global().start_profile(name, description);
  }

  /*!
   * Deconstructs the scoped heap profiler and thereby stopping the heapprofile.
   */
  inline ~ScopedHeapProfiler() {
    HeapProfiler::global().stop_profile();
  }
};

} // namespace kaminpar::heap_profiler
