cmake_minimum_required(VERSION 3.21)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

include(FetchContent)
include(CheckCXXCompilerFlag)

project(KaMinPar
        DESCRIPTION "Shared-memory and distributed-memory Graph Partitioner"
        LANGUAGES C CXX)

set(PROJECT_VENDOR "Daniel Seemaier")
set(PROJECT_CONTACT "daniel.seemaier@kit.edu")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

################################################################################
## Options                                                                    ##
################################################################################

# Control what to build
#######################
option(KAMINPAR_BUILD_TESTS "Build unit tests" ON)
option(KAMINPAR_BUILD_DISTRIBUTED "Build distributed partitioner." OFF)
option(KAMINPAR_BUILD_APPS "Build binaries." ON)
option(KAMINPAR_BUILD_BENCHMARKS "Build benchmark binaries." OFF)
option(KAMINPAR_BUILD_TOOLS "Build tool binaries." OFF)

option(KAMINPAR_BUILD_EXPERIMENTAL_FEATURES "Include experimental features in the build. This might increase compile times drastically." OFF)

# Control how to build
######################
option(KAMINPAR_ENABLE_HEAP_PROFILING "Profile and output heap memory usage." OFF)
option(KAMINPAR_ENABLE_PAGE_PROFILING "Profile pages allocated via mmap." OFF)
option(KAMINPAR_ENABLE_STATISTICS "Generate and output detailed statistics." OFF)
option(KAMINPAR_ENABLE_TIMERS "Measure running times. Must be set to 'OFF' if the library interface is used from multiple threads simulatinously." ON)
option(KAMINPAR_ENABLE_TIMER_BARRIERS "Add additional MPI_Barrier() instructions for more accurate time measurements." ON)

option(KAMINPAR_ENABLE_THP "Use transparent huge pages for large memory allocations (Linux only)." ON)

option(KAMINPAR_BUILD_WITH_ASAN "Enable address sanitizer." OFF)
option(KAMINPAR_BUILD_WITH_UBSAN "Enable undefined behaviour sanitizer." OFF)
option(KAMINPAR_BUILD_WITH_MTUNE_NATIVE "Build with -mtune=native." ON)
option(KAMINPAR_BUILD_WITH_CCACHE "Use ccache to build." ON)
option(KAMINPAR_BUILD_WITH_DEBUG_SYMBOLS "Always build with debug symbols, even in Release mode." ON)
option(KAMINPAR_BUILD_WITH_MTKAHYPAR "If Mt-KaHyPar can be found, build the Mt-KaHyPar initial partitioner." OFF)
option(KAMINPAR_BUILD_WITH_GROWT "Build the shared-memory partitioner with Growt." ON)
option(KAMINPAR_BUILD_WITH_SPARSEHASH "Build with Google Sparsehash." ON)
option(KAMINPAR_BUILD_WITH_PG "Build with the -pg option for profiling." OFF)
option(KAMINPAR_BUILD_WITH_BACKWARD "Build with backward-cpp for stack traces (distributed partitioner only)." OFF)

# Control data type sizes
#########################
# These IDs refer to the shared-memory partitioner + local IDs of the distributed partitioner
option(KAMINPAR_64BIT_IDS "Use 64 bits for node and edge IDs." OFF)
option(KAMINPAR_64BIT_EDGE_IDS "Use 64 bits for edge IDs." OFF)
option(KAMINPAR_64BIT_NODE_IDS "Use 64 bits for node IDs." OFF)

# Node and edge weights for the shared-memory partitioner (+ used as initial partitioner of the distributed partitioner)
option(KAMINPAR_64BIT_WEIGHTS "Use 64 bit for node and edge weights." OFF)

# Local node and edge weights for the distributed partitioner; should be 64 bit when using DMGP
option(KAMINPAR_64BIT_LOCAL_WEIGHTS "Use 64 bit for local node and edge weights." OFF)

# The distributed partitioner requires 64 bit node and edge weights for the coarsest graph, 
# which is copied to each PE and build with data types of the shared-memory partitioner.
# Thus, force 64 bit weights for the shared-memory partitioner in this case.
if (KAMINPAR_BUILD_DISTRIBUTED)
    message(STATUS "Distributed build: enabling 64 bit weights.")
    set(KAMINPAR_64BIT_WEIGHTS ON)
endif ()

# Control graph compression options
###################################
option(KAMINPAR_COMPRESSION_EDGE_WEIGHTS "Whether to compress edge weights." ON)
option(KAMINPAR_COMPRESSION_HIGH_DEGREE_ENCODING "Use high-degree encoding for the compressed graph." ON)
option(KAMINPAR_COMPRESSION_INTERVAL_ENCODING "Use interval encoding for the compressed graph." ON)
option(KAMINPAR_COMPRESSION_RUN_LENGTH_ENCODING "Use run-length encoding for the compressed graph." OFF)
option(KAMINPAR_COMPRESSION_STREAMVBYTE_ENCODING "Use StreamVByte encoding for the compressed graph." OFF)
option(KAMINPAR_COMPRESSION_FAST_DECODING "Use fast decoding for the compressed graph." OFF)

if (KAMINPAR_COMPRESSION_RUN_LENGTH_ENCODING AND KAMINPAR_COMPRESSION_STREAM_ENCODING)
    message(FATAL_ERROR "Either run-length or StreamVByte encoding can be used for varints but not both.")
endif ()

if (KAMINPAR_64BIT_NODE_IDS AND KAMINPAR_COMPRESSION_STREAM_ENCODING)
    message(FATAL_ERROR "StreamVByte encoding cannot be used with 64-bit NodeIDs.")
endif ()

if (KAMINPAR_COMPRESSION_EDGE_WEIGHTS AND KAMINPAR_COMPRESSION_RUN_LENGTH_ENCODING)
    message(FATAL_ERROR "Run-length encoding cannot be used together with compressed edge weights.")
endif ()

################################################################################
## Declare dependencies                                                       ##
################################################################################

set(KAMINPAR_ASSERTION_LEVEL "light" CACHE STRING "Assertion level.")
set_property(CACHE KAMINPAR_ASSERTION_LEVEL PROPERTY STRINGS none light normal heavy)
message(STATUS "KAssertion level: ${KAMINPAR_ASSERTION_LEVEL}")

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set warning flags
list(APPEND KAMINPAR_WARNING_FLAGS
    "-W"
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wno-unused-local-typedefs"
    )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    list(APPEND KAMINPAR_WARNING_FLAGS
        "-Wextra-semi"
        "-fcolor-diagnostics"
        "-Wdeprecated"
        )
endif ()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(APPEND KAMINPAR_WARNING_FLAGS
        "-Wsuggest-override"
        "-fdiagnostics-color=always"
        "-Wcast-qual"
        "-Winit-self"
        "-Woverloaded-virtual"
        "-Wredundant-decls"
        )
endif ()

# Build experimental features that increase compile times
if (KAMINPAR_BUILD_EXPERIMENTAL_FEATURES)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_EXPERIMENTAL")
endif ()

# Always enable Debug symbols (including in Release mode)
if (KAMINPAR_BUILD_WITH_DEBUG_SYMBOLS)
    add_compile_options(-g)
endif ()

# Set compile flags
set(CMAKE_REQUIRED_FLAGS -Werror) # otherwise the check fails for Apple Clang 
check_cxx_compiler_flag(-msse4.1 COMPILER_SUPPORTS_MSSE41)
if (COMPILER_SUPPORTS_MSSE41)
    add_compile_options(-msse4.1)
endif ()
check_cxx_compiler_flag(-mcx16 COMPILER_SUPPORTS_MCX16)
if (COMPILER_SUPPORTS_MCX16)
    add_compile_options(-mcx16)
endif ()

if (KAMINPAR_BUILD_WITH_MTUNE_NATIVE) 
    add_compile_options(-mtune=native -march=native)
endif ()

if (KAMINPAR_BUILD_WITH_ASAN) 
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

if (KAMINPAR_BUILD_WITH_UBSAN) 
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif ()

if (KAMINPAR_BUILD_WITH_PG)
    add_compile_options(-pg)
endif ()

# Pass CMake options to code
if (KAMINPAR_ENABLE_STATISTICS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_STATISTICS")
    message(STATUS "Statistics: enabled")
else ()
    message(STATUS "Statistics: disabled")
endif ()

if (KAMINPAR_ENABLE_HEAP_PROFILING)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_HEAP_PROFILING")
    message(STATUS "Heap Profiling: enabled")
else ()
    message(STATUS "Heap Profiling: disabled")
endif ()

if (KAMINPAR_ENABLE_PAGE_PROFILING)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_PAGE_PROFILING")
    message(STATUS "Page Profiling: enabled")
else ()
    message(STATUS "Page Profiling: disabled")
endif ()

if (KAMINPAR_ENABLE_TIMERS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_TIMERS")
    message(STATUS "Timers: enabled")
else ()
    message(STATUS "Timers: disabled")
endif ()

if (KAMINPAR_ENABLE_TIMER_BARRIERS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_TIMER_BARRIERS")
    message(STATUS "Timer barriers: enabled")
else ()
    message(STATUS "Timer barriers: disabled")
endif ()

if (KAMINPAR_ENABLE_THP)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_THP")
    message(STATUS "Huge pages: enabled")
else ()
    message(STATUS "Huge pages: disabled")
endif ()

message(STATUS "Graph compression summary:")

if (KAMINPAR_COMPRESSION_EDGE_WEIGHTS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_COMPRESSION_EDGE_WEIGHTS")
    message("  Compressed edge weights: enabled")
else ()
    message("  Compressed edge weights: disabled")
endif ()

if (KAMINPAR_COMPRESSION_HIGH_DEGREE_ENCODING)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_COMPRESSION_HIGH_DEGREE_ENCODING")
    message("  High-degree encoding: enabled")
else ()
    message("  High-degree encoding: disabled")
endif ()

if (KAMINPAR_COMPRESSION_INTERVAL_ENCODING)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_COMPRESSION_INTERVAL_ENCODING")
    message("  Interval encoding: enabled")
else ()
    message("  Interval encoding: disabled")
endif ()

if (KAMINPAR_COMPRESSION_RUN_LENGTH_ENCODING)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_COMPRESSION_RUN_LENGTH_ENCODING")
    message("  Run-length encoding: enabled")
else ()
    message("  Run-length encoding: disabled")
endif ()

if (KAMINPAR_COMPRESSION_STREAMVBYTE_ENCODING)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_COMPRESSION_STREAMVBYTE_ENCODING")
    message("  StreamVByte encoding: enabled")
else ()
    message("  StreamVByte encoding: disabled")
endif ()

if (KAMINPAR_COMPRESSION_FAST_DECODING)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_COMPRESSION_FAST_DECODING")
    add_compile_options(-mbmi2)
    message("  Fast decoding: enabled")
else ()
    message("  Fast decoding: disabled")
endif ()

if (KAMINPAR_64BIT_NODE_IDS OR KAMINPAR_64BIT_IDS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_NODE_IDS")
    set(KAMINPAR_SHM_NODE_ID_STR "std::uint64_t")
else ()
    set(KAMINPAR_SHM_NODE_ID_STR "std::uint32_t")
endif ()

if (KAMINPAR_64BIT_EDGE_IDS OR KAMINPAR_64BIT_IDS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_EDGE_IDS")
    set(KAMINPAR_SHM_EDGE_ID_STR "std::uint64_t")
else ()
    set(KAMINPAR_SHM_EDGE_ID_STR "std::uint32_t")
endif ()

if (KAMINPAR_64BIT_WEIGHTS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_WEIGHTS")
    set(KAMINPAR_SHM_WEIGHT_STR "std::int64_t")
else () 
    set(KAMINPAR_SHM_WEIGHT_STR "std::int32_t")
endif ()

if (KAMINPAR_64BIT_LOCAL_WEIGHTS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_LOCAL_WEIGHTS")
    set(KAMINPAR_DIST_WEIGHT_STR "std::int64_t")
else ()
    set(KAMINPAR_DIST_WEIGHT_STR "std::int32_t")
endif ()

message(STATUS "Data type summary:")
message("  {shm, dist}::NodeID: ${KAMINPAR_SHM_NODE_ID_STR}")
message("  {shm, dist}::EdgeID: ${KAMINPAR_SHM_EDGE_ID_STR}")
message("  shm::{Node, Edge}Weight: ${KAMINPAR_SHM_WEIGHT_STR}")
message("  {dist::Global{Node, Edge}ID: std::uint64_t")
message("  dist::Global{Node, Edge}Weight: std::int64_t")
message("  dist::{Node, Edge}Weight: ${KAMINPAR_DIST_WEIGHT_STR}")

################################################################################
## Search and fetch dependencies                                              ##
################################################################################

# Google Sparsehash 
if (KAMINPAR_BUILD_WITH_SPARSEHASH)
    find_package(Sparsehash REQUIRED)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_SPARSEHASH_FOUND")
endif ()

if (KAMINPAR_BUILD_WITH_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif ()
endif ()

if (KAMINPAR_BUILD_WITH_GROWT)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_USES_GROWT")
endif ()

if (TRUE) 
    add_subdirectory(external/growt EXCLUDE_FROM_ALL)
    add_library(growt INTERFACE)
    target_include_directories(growt SYSTEM INTERFACE "external/growt")
endif ()

if (KAMINPAR_BUILD_DISTRIBUTED)
    # MPI
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
    find_package(MPI)
    if (NOT MPI_FOUND) 
        message(WARNING "MPI not available: cannot build the distributed partitioner")
        set(KAMINPAR_BUILD_DISTRIBUTED OFF)
    endif ()

    if (KAMINPAR_BUILD_WITH_BACKWARD)
        add_subdirectory(external/bakward-mpi EXCLUDE_FROM_ALL)
    endif ()
endif ()

if (KAMINPAR_ASSERTION_LEVEL STREQUAL "none")
    set(KASSERT_ASSERTION_LEVEL 0)
elseif (KAMINPAR_ASSERTION_LEVEL STREQUAL "light")
    set(KASSERT_ASSERTION_LEVEL 10)
elseif (KAMINPAR_ASSERTION_LEVEL STREQUAL "normal")
    set(KASSERT_ASSERTION_LEVEL 30)
elseif (KAMINPAR_ASSERTION_LEVEL STREQUAL "heavy")
    set(KASSERT_ASSERTION_LEVEL 40)
else ()
    message(WARNING "Invalid assertion level: ${KAMINPAR_ASSERTION_LEVEL}")
endif ()

# Add KAssert
add_subdirectory(external/kassert EXCLUDE_FROM_ALL)

# If we can find Mt-KaHyPar, make it available for initial partitioning and refinement
if (KAMINPAR_BUILD_WITH_MTKAHYPAR)
    find_library(LIB_MTKAHYPAR_GRAPH mtkahypar)
    if (NOT LIB_MTKAHYPAR_GRAPH)
        message(STATUS "Mt-KaHyPar initial partitioning not available: library could not be found on this system")
        set(KAMINPAR_BUILD_WITH_MTKAHYPAR OFF)
    else ()
        message(STATUS "Found Mt-KaHyPar at ${LIB_MTKAHYPAR_GRAPH}")
    endif ()
endif ()

# Fetch minimal KaGen for graph IO
if ((KAMINPAR_BUILD_DISTRIBUTED AND KAMINPAR_BUILD_APPS) OR KAMINPAR_BUILD_BENCHMARKS)
    set(KAGEN_NODEPS ON CACHE BOOL "")
    set(KAGEN_BUILD_APPS OFF CACHE BOOL "")
    set(KAGEN_BUILD_EXAMPLES OFF CACHE BOOL "")
    set(KAGEN_BUILD_TESTS OFF CACHE BOOL "")
    add_subdirectory(external/KaGen EXCLUDE_FROM_ALL)
endif ()

################################################################################
## Add targets in subdirectories                                              ##
################################################################################

# Start include paths on project root
include_directories(${PROJECT_SOURCE_DIR})

# Shared memory components
add_subdirectory(kaminpar-common)
add_subdirectory(kaminpar-shm)

# Distributed components
if (KAMINPAR_BUILD_DISTRIBUTED)
    add_subdirectory(kaminpar-mpi)
    add_subdirectory(kaminpar-dist)
endif ()

# Binaries
add_subdirectory(kaminpar-cli)

if (KAMINPAR_BUILD_APPS)
    add_subdirectory(apps)
endif ()

# Unit tests
if (KAMINPAR_BUILD_TESTS)
    add_subdirectory(external/googletest EXCLUDE_FROM_ALL SYSTEM)

    enable_testing()
    add_subdirectory(tests)
endif ()

################################################################################

add_library(KaMinPar::KaMinPar ALIAS kaminpar_shm)
add_library(KaMinPar::KaMinParCLI ALIAS kaminpar_cli)

if (KAMINPAR_BUILD_DISTRIBUTED)
    add_library(KaMinPar::dKaMinPar ALIAS kaminpar_dist)
    add_library(KaMinPar::dKaMinParCLI ALIAS kaminpar_dist_cli)
endif ()

