/*******************************************************************************
 * This file overwrites the memory allocation operations of libc with operations that additionally
 * invoke the heap profiler.
 *
 * @file:   libc_memory_override.h
 * @author: Daniel Salwasser
 * @date:   22.10.2023
 ******************************************************************************/
#pragma once

#include <cstddef>

namespace kaminpar::heap_profiler {

/*!
 * Allocates size bytes of uninitialized memory. The allocation request is directly forwarded to
 * malloc and thus not captured by the heap profiler.
 *
 * @param size The number of bytes to allocate.
 *
 * @return Returns the pointer to the beginning of newly allocated memory on success, otherwise a
 * null pointer.
 */
void *std_malloc(std::size_t size);

/*!
 * Deallocates the memory previously allocated by std_malloc.
 *
 * @param ptr The pointer to the memory to be deallocated.
 */
void std_free(void *ptr);

} // namespace kaminpar::heap_profiler
