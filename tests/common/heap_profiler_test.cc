#include <iostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kaminpar-common/heap_profiler.h"

// Allocate memory such that the compiler does not optimise it away.
#define ALLOC_ARR(name, size)                                                                      \
  char *name = new char[size];                                                                     \
  volatile auto name##_copy = *((char *)name);

namespace kaminpar::heap_profiler {

TEST(HeapProfilerTest, NewArrayOperator) {
  const std::size_t size = 1024;

  HeapProfiler::global().enable();

  ALLOC_ARR(array, size)
  delete[] array;

  HeapProfiler::global().disable();

#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  EXPECT_EQ(size, HeapProfiler::global().get_alloc());
  EXPECT_EQ(size, HeapProfiler::global().get_max_alloc());
  EXPECT_EQ(1, HeapProfiler::global().get_allocs());
  EXPECT_EQ(1, HeapProfiler::global().get_frees());
#endif // KAMINPAR_ENABLE_HEAP_PROFILING
}

TEST(HeapProfilerTest, MaxAllocTest) {
  HeapProfiler::global().enable();

  ALLOC_ARR(array1, 1024);
  delete[] array1;

#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  EXPECT_EQ(1024, HeapProfiler::global().get_max_alloc());
#endif // KAMINPAR_ENABLE_HEAP_PROFILING

  ALLOC_ARR(array2, 2048);
  delete[] array2;
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  EXPECT_EQ(2048, HeapProfiler::global().get_max_alloc());
#endif // KAMINPAR_ENABLE_HEAP_PROFILING

  ALLOC_ARR(array3, 128);
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  EXPECT_EQ(2048, HeapProfiler::global().get_max_alloc());
#endif // KAMINPAR_ENABLE_HEAP_PROFILING

  ALLOC_ARR(array4, 4096);
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  EXPECT_EQ(4224, HeapProfiler::global().get_max_alloc());
#endif // KAMINPAR_ENABLE_HEAP_PROFILING
  delete[] array3;
  delete[] array4;

  HeapProfiler::global().disable();
}

} // namespace kaminpar::heap_profiler
