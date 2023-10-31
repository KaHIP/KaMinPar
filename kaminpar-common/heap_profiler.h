/*******************************************************************************
 * Heap profiler to measure heap memory usage.
 *
 * @file:   heap_profiler.h
 * @author: Daniel Salwasser
 * @date:   21.10.2023
 ******************************************************************************/
#pragma once

#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifdef KAMINPAR_ENABLE_HEAP_PROFILING

#include "kaminpar-common/libc_memory_override.h"

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
#define ENABLE_HEAP_PROFILER() kaminpar::heap_profiler::HeapProfiler::global().enable()
#define DISABLE_HEAP_PROFILER() kaminpar::heap_profiler::HeapProfiler::global().disable()
#define PRINT_HEAP_PROFILE(out)                                                                    \
  kaminpar::heap_profiler::HeapProfiler::global().print_heap_profile(out)

#else

#define START_HEAP_PROFILER(...)
#define STOP_HEAP_PROFILER()
#define SCOPED_HEAP_PROFILER(...)
#define ENABLE_HEAP_PROFILER()
#define DISABLE_HEAP_PROFILER()
#define PRINT_HEAP_PROFILE(...)

#endif

namespace kaminpar::heap_profiler {

/*!
 * A minimal allocator that uses memory allocation functions which bypass the heap profiler.
 *
 * This is required for allocations inside the heap profiler, otherwise a memory allocation would
 * lead to an infinite recursion.
 */
template <class T> struct NoProfilAllocator {
  using value_type = T;

  NoProfilAllocator() noexcept {}
  template <class U> NoProfilAllocator(const NoProfilAllocator<U> &) noexcept {}

  template <class U> bool operator==(const NoProfilAllocator<U> &) const noexcept {
    return true;
  }
  template <class U> bool operator!=(const NoProfilAllocator<U> &) const noexcept {
    return false;
  }

  T *allocate(const size_t n) const;
  void deallocate(T *const p, size_t) const noexcept;

  T *construct() const;
  void destruct(T *const t) const;
};

class ScopedHeapProfiler;

struct HeapProfileTreeStats {
  std::size_t max_len;
  std::size_t max_alloc;
  std::size_t max_allocs;
  std::size_t max_frees;
};

/*!
 * A hierarchical heap profiler to measure dynamic memory allocation of the program.
 *
 * The memory allocation operations of libc are overridden to additionally call the global heap
 * profiler on each allocation and deallocation request.
 */
class HeapProfiler {
private:
  static constexpr std::string_view kAllocSizeTitle = "Alloc (mb)";
  static constexpr std::string_view kAllocsTitle = "Allocs";
  static constexpr std::string_view kFreesTitle = "Frees";

  static constexpr std::string_view kBranch = "|-- ";
  static constexpr std::string_view kTailBranch = "`-- ";
  static constexpr std::string_view kTailEdge = "    ";
  static constexpr std::string_view kNameDel = ": ";
  static constexpr std::size_t kBranchLength = 4;
  static constexpr char kPadding = '.';

  static std::string to_megabytes(std::size_t bytes) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << (bytes / (float)(1024 * 1024));
    return stream.str();
  }

  struct HeapProfileTreeNode {
    std::string_view name;
    std::string description;
    std::vector<HeapProfileTreeNode *, NoProfilAllocator<HeapProfileTreeNode *>> children;
    HeapProfileTreeNode *parent;

    std::size_t allocs;
    std::size_t frees;
    std::size_t alloc_size;

    template <typename Allocator> void free(Allocator allocator) {
      for (HeapProfileTreeNode *child : children) {
        child->free(allocator);
        allocator.destruct(child);
      }
    }
  };

  struct HeapProfileTree {
    HeapProfileTreeNode root;
    HeapProfileTreeNode *currentNode = &root;
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
   * @param description The description of the profile to start.
   */
  void start_profile(std::string_view name, std::string description);

  /*!
   * Stops the current profile and sets the new current profile to the parent profile.
   */
  void stop_profile();

  /*!
   * Starts a scoped heap profile and returns the associated object.
   *
   * @param name The name of the profile to start.
   * @param description The description of the profile to start.
   */
  ScopedHeapProfiler start_scoped_profile(std::string_view name, std::string description);

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
   * Prints information about the heap profile to the output stream.
   *
   * @param out The output stream to write to.
   */
  void print_heap_profile(std::ostream &out);

  std::size_t get_max_alloc();
  std::size_t get_alloc();
  std::size_t get_allocs();
  std::size_t get_frees();

private:
  bool _enabled = false;
  std::string_view _name;

  std::mutex _mutex;
  NoProfilAllocator<HeapProfileTreeNode> _node_allocator;
  std::unordered_map<
      const void *,
      std::size_t,
      std::hash<const void *>,
      std::equal_to<const void *>,
      NoProfilAllocator<std::pair<const void *const, std::size_t>>>
      _address_map;
  HeapProfileTree _tree;

  std::size_t _total_alloc;
  std::size_t _total_free;
  std::size_t _max_alloc;
  std::size_t _allocs;
  std::size_t _frees;

  HeapProfileTreeStats calculate_stats(const HeapProfileTreeNode &node);

  void print_heap_tree_node(
      std::ostream &out,
      const HeapProfileTreeNode &node,
      HeapProfileTreeStats stats,
      std::size_t depth = 0,
      bool last = false
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
