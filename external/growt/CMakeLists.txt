################################################################################
# CMakeLists.txt
#
# Root CMake build script for growt.
#
# Part of Project growt - https://github.com/TooBiased/growt.git
#
# Copyright (C) 2015-2016 Tobias Maier <t.maier@kit.edu>
#
# All rights reserved. Published under the BSD-2 license in the LICENSE file.
################################################################################
cmake_minimum_required (VERSION 3.21)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/misc/cmake)
project(concurrent-growtable)

set(GROWT_BUILD_MODE RELEASE CACHE STRING
  "the build mode of the current tests (controls optimizations and symbols)!")
set_property(CACHE GROWT_BUILD_MODE PROPERTY STRINGS RELEASE DEBUG REL_W_SYMBOLS )

set(GROWT_ALLOCATOR ALIGNED CACHE STRING
  "Specifies the used allocator (only relevant for our tables)!")
set_property(CACHE GROWT_ALLOCATOR PROPERTY STRINGS ALIGNED)

set(GROWT_ALLOCATOR_POOL_SIZE 2 CACHE STRING
  "Size of preallocated memory pool (only relevant for pool allocators)!")
if (NOT GROWT_ALLOCATOR_POOL_SIZE MATCHES "^[0-9]+$")
  message(FATAL_ERROR "GROWT_ALLOCATOR_POOL_SIZE must be a numeric argument")
endif()

set(GROWT_HASHFCT XXH3 CACHE STRING
  "Changes the used hash function if XXHASH is not available, MURMUR2 is used as backoff!")
set_property(CACHE GROWT_HASHFCT PROPERTY STRINGS XXH3 XXHASH MURMUR2 CRC)

include_directories(.)

find_package(Threads REQUIRED)

if ((GROWT_HASHFCT STREQUAL XXHASH) OR
    (GROWT_HASHFCT STREQUAL XXH3))
  message(STATUS "Looking for xxhash")

  find_path(XXHASH_DIR xxhash/xxhash.h
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/misc/submodules)

  if (XXHASH_DIR)
    include_directories(${XXHASH_DIR}/xxhash)
    message(STATUS "Looking for xxhash -- found")
  else()
    message(STATUS "Looking for xxhash -- not found")
    message("Cannot find xxHash. Use MURMUR2 instead.")
    set(GROWT_HASHFCT MURMUR2)
  endif()
endif()

configure_file ("misc/growt_config.h.in" "${PROJECT_BINARY_DIR}/growt_config.h")
include_directories(${PROJECT_BINARY_DIR})

