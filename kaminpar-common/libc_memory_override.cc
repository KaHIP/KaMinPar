/*******************************************************************************
 * This file overwrites the memory allocation operations of libc with operations that additionally
 * invoke the heap profiler.
 *
 * @file:   libc_memory_override.cc
 * @author: Daniel Salwasser
 * @date:   22.10.2023
 ******************************************************************************/
#include "kaminpar-common/libc_memory_override.h"

#include <cstdlib>

#include "kaminpar-common/heap_profiler.h"

#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
#ifdef __GLIBC__
extern "C" {

using kaminpar::heap_profiler::HeapProfiler;

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_free(void *);
extern void *__libc_memalign(size_t, size_t);
extern void *__libc_valloc(size_t);
extern void *__libc_pvalloc(size_t);
extern void *__libc_realloc(void *, size_t);

void *malloc(size_t size) {
  void *ptr = __libc_malloc(size);
  HeapProfiler::global().record_alloc(ptr, size);
  return ptr;
};

void *calloc(size_t size, size_t n) {
  void *ptr = __libc_calloc(size, n);
  HeapProfiler::global().record_alloc(ptr, size * n);
  return ptr;
}

void *realloc(void *p, size_t newsize) {
  void *ptr = __libc_realloc(p, newsize);
  HeapProfiler::global().record_free(p);
  HeapProfiler::global().record_alloc(ptr, newsize);
  return ptr;
}

void free(void *p) {
  __libc_free(p);
  HeapProfiler::global().record_free(p);
}

void *aligned_alloc(size_t alignment, size_t size) {
  // Since glibc does not define aligned_alloc as a weak symbol to e.g. __libc_aligned_alloc, unlike
  // other functions, __libc_memalign is called instead with a check for valid alignment.
  bool is_power_of_2 = (alignment & (alignment - 1)) == 0;
  if (!is_power_of_2 || alignment == 0) {
    errno = EINVAL;
    return 0;
  }

  void *ptr = __libc_memalign(alignment, size);
  HeapProfiler::global().record_alloc(ptr, size);
  return ptr;
}

void *memalign(size_t alignment, size_t size) {
  void *ptr = __libc_memalign(alignment, size);
  HeapProfiler::global().record_alloc(ptr, size);
  return ptr;
}

void *valloc(size_t size) {
  void *ptr = __libc_valloc(size);
  HeapProfiler::global().record_alloc(ptr, size);
  return ptr;
}

void *pvalloc(size_t size) {
  void *ptr = __libc_pvalloc(size);
  HeapProfiler::global().record_alloc(ptr, size);
  return ptr;
}

#ifdef KAMINPAR_ENABLE_PAGE_PROFILING
extern void *__mmap(void *, size_t, int, int, int, off_t);
extern int __munmap(void *, size_t);

void *mmap(void *addr, size_t len, int prot, int flags, int fd, __off_t offset) {
  void *ptr = __mmap(addr, len, prot, flags, fd, offset);
  HeapProfiler::global().record_alloc(addr, len);
  return ptr;
}

int munmap(void *addr, size_t len) {
  int return_value = __munmap(addr, len);
  HeapProfiler::global().record_free(addr);
  return return_value;
}
#endif
}
#else
#error Heap profiling is only supported for systems that are using glibc.
#endif
#endif

namespace kaminpar::heap_profiler {

void *std_malloc(std::size_t size) {
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  return __libc_malloc(size);
#else
  return std::malloc(size);
#endif
}

void std_free(void *ptr) {
#ifdef KAMINPAR_ENABLE_HEAP_PROFILING
  __libc_free(ptr);
#else
  std::free(ptr);
#endif
}

} // namespace kaminpar::heap_profiler
