cmake_minimum_required(VERSION 3.21)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

include(FetchContent)
include(CheckCXXCompilerFlag)

project(KaMinPar
        DESCRIPTION "Shared-memory and distributed graph partitioner for large k partitioning."
        LANGUAGES C CXX)

set(PROJECT_VENDOR "Daniel Seemaier")
set(PROJECT_CONTACT "daniel.seemaier@kit.edu")
set(CMAKE_CXX_STANDARD 17)

################################################################################
## Options                                                                    ##
################################################################################

# Control what to build
#######################
option(KAMINPAR_BUILD_TESTS "Build unit tests" ON)
option(KAMINPAR_BUILD_DISTRIBUTED "Build distributed partitioner." OFF)
option(KAMINPAR_BUILD_APPS "Build binaries." ON)
option(KAMINPAR_BUILD_BENCHMARKS "Build benchmark binaries." OFF)

option(KAMINPAR_BUILD_EXPERIMENTAL_FEATURES "Include experimental features in the build. This might increase compile times drastically." OFF)

# Control how to build
######################
option(KAMINPAR_ENABLE_STATISTICS "Generate and output detailed statistics." ON)
option(KAMINPAR_ENABLE_TIMERS "Measure running times. Must be set to 'OFF' if the library interface is used from multiple threads simulatinously." ON)
option(KAMINPAR_ENABLE_TIMER_BARRIERS "Add additional MPI_Barrier() instructions for more accurate time measurements." ON)

option(KAMINPAR_BUILD_WITH_ASAN "Enable address sanitizer." OFF)
option(KAMINPAR_BUILD_WITH_UBSAN "Enable undefined behaviour sanitizer." OFF)
option(KAMINPAR_BUILD_WITH_MTUNE_NATIVE "Build with -mtune=native." ON)
option(KAMINPAR_BUILD_WITH_CCACHE "Use ccache to build." ON)
option(KAMINPAR_BUILD_WITH_DEBUG_SYMBOLS "Always build with debug symbols, even in Release mode." ON)
option(KAMINPAR_BUILD_WITH_MTKAHYPAR "If Mt-KaHyPar can be found, build the Mt-KaHyPar initial partitioner." OFF)
option(KAMINPAR_BUILD_WITH_PG "Build with the -pg option for profiling." OFF)

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
    set(KAMINPAR_64BIT_WEIGHTS ON)
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
        "-Wnoexcept"
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
    add_compile_options(-g -g3)
endif ()

# Set compile flags 
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
    message(STATIS "Statistics: disabled")
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
    set(KAMINPAR_SHM_NODE_WEIGHT_STR "std::int64_t")
    set(KAMINPAR_SHM_EDGE_WEIGHT_STR "std::int64_t")
else () 
    set(KAMINPAR_SHM_NODE_WEIGHT_STR "std::int32_t")
    set(KAMINPAR_SHM_EDGE_WEIGHT_STR "std::int32_t")
endif ()

if (KAMINPAR_64BIT_LOCAL_WEIGHTS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_LOCAL_WEIGHTS")
    set(KAMINPAR_DIST_NODE_WEIGHT_STR "std::int64_t")
    set(KAMINPAR_DIST_EDGE_WEIGHT_STR "std::int64_t")
else ()
    set(KAMINPAR_DIST_NODE_WEIGHT_STR "std::int32_t")
    set(KAMINPAR_DIST_EDGE_WEIGHT_STR "std::int32_t")
endif ()

message(STATUS "Data type summary:")
message("  {shm, dist}::NodeID    = ${KAMINPAR_SHM_NODE_ID_STR} | {shm, dist}::EdgeID    = ${KAMINPAR_SHM_EDGE_ID_STR}")
message("  dist::GlobalNodeID     = std::uint64_t | dist::GlobalEdgeID     = std::uint64_t")
message("  shm::NodeWeight        =  ${KAMINPAR_SHM_NODE_WEIGHT_STR} | shm::EdgeWeight        =  ${KAMINPAR_SHM_EDGE_WEIGHT_STR}")
message("  dist::NodeWeight       =  ${KAMINPAR_DIST_NODE_WEIGHT_STR} | dist::EdgeWeight       =  ${KAMINPAR_DIST_EDGE_WEIGHT_STR}")
message("  dist::GlobalNodeWeight =  std::int64_t | dist::GlobalEdgeWeight =  std::int64_t")
message("  {shm, dist}::BlockID   = std::uint32_t")
message("  shm::BlockWeight       =  ${KAMINPAR_SHM_NODE_WEIGHT_STR}")
message("  dist::BlockWeight      =  std::int64_t")

################################################################################
## Search and fetch dependencies                                              ##
################################################################################

if (KAMINPAR_BUILD_WITH_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif ()
endif ()

if (KAMINPAR_BUILD_DISTRIBUTED)
    # MPI
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
    find_package(MPI)
    if (NOT MPI_FOUND) 
        message(WARNING "MPI not available: cannot build the distributed partitioner")
        set(KAMINPAR_BUILD_DISTRIBUTED OFF)
    endif ()

    # Growt (needs -mcx16, i.e., does not work on ARM)
    check_cxx_compiler_flag(-mcx16 COMPILER_SUPPORTS_MCX16)
    if (COMPILER_SUPPORTS_MCX16) 
        add_compile_options(-mcx16)
    else ()
        message(WARNING "-mcx16 flag not supported by the compiler: cannot build the distributed partitioner")
        set(KAMINPAR_BUILD_DISTRIBUTED OFF)
    endif()

    add_subdirectory(external/growt EXCLUDE_FROM_ALL)
    add_library(growt INTERFACE)
    target_include_directories(growt SYSTEM INTERFACE "external/growt")

    # Google Sparsehash 
    find_package(Sparsehash REQUIRED)
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
    add_subdirectory(external/googletest EXCLUDE_FROM_ALL)

    enable_testing()
    add_subdirectory(tests)
endif ()

################################################################################

add_library(KaMinPar::KaMinPar ALIAS kaminpar_shm)
add_library(KaMinPar::KaMinParCLI ALIAS kaminpar_cli)
add_library(KaMinPar::KaMinParCLI11 ALIAS kaminpar_cli) # deprecated

if (KAMINPAR_BUILD_DISTRIBUTED)
    add_library(KaMinPar::dKaMinPar ALIAS kaminpar_dist)
    add_library(KaMinPar::dKaMinParCLI ALIAS kaminpar_dist_cli)
    add_library(KaMinPar::dKaMinParCLI11 ALIAS kaminpar_dist_cli) # deprecated
endif ()

