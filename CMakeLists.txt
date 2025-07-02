cmake_minimum_required(VERSION 3.21)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

project(
  KaMinPar
  VERSION 3.6.0
  DESCRIPTION "Shared-memory and distributed-memory Graph Partitioner"
  HOMEPAGE_URL "https://github.com/KaHIP/KaMinPar"
  LANGUAGES C CXX)

set(PROJECT_VENDOR "Daniel Seemaier")
set(PROJECT_CONTACT "daniel.seemaier@kit.edu")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

################################################################################
## Declare Options                                                            ##
################################################################################

# Control what to build
#######################
option(KAMINPAR_BUILD_APPS "Build binaries." ON)
option(KAMINPAR_BUILD_DISTRIBUTED "Build distributed partitioner." OFF)
option(KAMINPAR_BUILD_TESTS "Build unit tests" OFF)
option(KAMINPAR_BUILD_TOOLS "Build tool binaries." OFF)
option(KAMINPAR_BUILD_BENCHMARKS "Build benchmark binaries." OFF)
option(KAMINPAR_BUILD_EXAMPLES "Build examples." OFF)

option(KAMINPAR_BUILD_EXPERIMENTAL_FEATURES "Include experimental features in the build. This might increase compile times drastically." OFF)

option(KAMINPAR_DOWNLOAD_TBB "Always download TBB even if it's available on the system." OFF)
option(KAMINPAR_DOWNLOAD_KASSERT "Always download kassert even if it's available on the system." OFF)
option(KAMINPAR_DOWNLOAD_KAGEN "Always download KaGen even if it's available on the system." OFF)
option(KAMINPAR_DOWNLOAD_MTKAHYPAR "Always download Mt-KaHyPar even if it's available on the system." OFF)
option(KAMINPAR_DOWNLOAD_GTEST "Always download GoogleTest even if it's available on the system." OFF)

# Control how to build
######################
option(KAMINPAR_ENABLE_HEAP_PROFILING "Profile and output heap memory usage." OFF)
option(KAMINPAR_ENABLE_PAGE_PROFILING "Profile pages allocated via mmap." OFF)
option(KAMINPAR_ENABLE_STATISTICS "Generate and output detailed statistics." OFF)
option(KAMINPAR_ENABLE_TIMERS "Measure running times. Must be set to 'OFF' if the library interface is used from multiple threads simulatinously." ON)
option(KAMINPAR_ENABLE_TIMER_BARRIERS "Add additional MPI_Barrier() instructions for more accurate time measurements." ON)

option(KAMINPAR_ENABLE_TBB_MALLOC "Use tbb malloc for (some) allocations." ON)
option(KAMINPAR_ENABLE_THP "Use transparent huge pages for large memory allocations (Linux only)." ON)

option(KAMINPAR_BUILD_WITH_ASAN "Enable address sanitizer." OFF)
option(KAMINPAR_BUILD_WITH_UBSAN "Enable undefined behaviour sanitizer." OFF)
option(KAMINPAR_BUILD_WITH_MTUNE_NATIVE "Build with -mtune=native." ON)
option(KAMINPAR_BUILD_WITH_CCACHE "Use ccache to build." ON)
option(KAMINPAR_BUILD_WITH_DEBUG_SYMBOLS "Always build with debug symbols, even in Release mode." ON)
option(KAMINPAR_BUILD_WITH_MTKAHYPAR "If Mt-KaHyPar can be found, build the Mt-KaHyPar initial partitioner." OFF)
option(KAMINPAR_BUILD_WITH_SPARSEHASH "Build with Google Sparsehash." ON)
option(KAMINPAR_BUILD_WITH_PG "Build with the -pg option for profiling." OFF)
option(KAMINPAR_BUILD_WITH_BACKWARD "Build with backward-cpp for stack traces (distributed partitioner only)." OFF)
option(KAMINPAR_BUILD_WITH_KASSERT "Use KASSERT for assertions. If disabled, the assertion level is ignored." ON)

# Control whether to install KaMinPar
#####################################
option(INSTALL_KAMINPAR "Install KaMinPar." ON)

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
if(KAMINPAR_BUILD_DISTRIBUTED)
  message(STATUS "Distributed build: enabling 64 bit weights.")
  set(KAMINPAR_64BIT_WEIGHTS ON)
endif()

# Control graph compression options
###################################
option(KAMINPAR_COMPRESSION_HIGH_DEGREE_ENCODING "Use high-degree encoding for the compressed graph." ON)
option(KAMINPAR_COMPRESSION_INTERVAL_ENCODING "Use interval encoding for the compressed graph." ON)
option(KAMINPAR_COMPRESSION_STREAMVBYTE_ENCODING "Use StreamVByte encoding for the compressed graph." OFF)
option(KAMINPAR_COMPRESSION_FAST_DECODING "Use a fast PEXT-based decoding routine for the compressed graph." OFF)

if(KAMINPAR_64BIT_NODE_IDS AND KAMINPAR_COMPRESSION_STREAM_ENCODING)
  message(FATAL_ERROR "StreamVByte encoding cannot be used with 64-bit NodeIDs.")
endif()

################################################################################
## Declare dependencies                                                       ##
################################################################################
include(CheckCXXCompilerFlag)

set(KAMINPAR_ASSERTION_LEVEL "light" CACHE STRING "Assertion level.")
set_property(
  CACHE KAMINPAR_ASSERTION_LEVEL
  PROPERTY STRINGS none light normal heavy
)
message(STATUS "KAssertion level: ${KAMINPAR_ASSERTION_LEVEL}")

if(KAMINPAR_ASSERTION_LEVEL STREQUAL "none")
  set(KAMINPAR_ASSERTION_LEVEL_CODE 0)
elseif(KAMINPAR_ASSERTION_LEVEL STREQUAL "light")
  set(KAMINPAR_ASSERTION_LEVEL_CODE 10)
elseif(KAMINPAR_ASSERTION_LEVEL STREQUAL "normal")
  set(KAMINPAR_ASSERTION_LEVEL_CODE 30)
elseif(KAMINPAR_ASSERTION_LEVEL STREQUAL "heavy")
  set(KAMINPAR_ASSERTION_LEVEL_CODE 40)
else()
  message(WARNING "Invalid assertion level: ${KAMINPAR_ASSERTION_LEVEL}")
endif()

set(KASSERT_ASSERTION_LEVEL ${KAMINPAR_ASSERTION_LEVEL_CODE})

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set warning flags
list(APPEND KAMINPAR_WARNING_FLAGS
  "-W"
  "-Wall"
  "-Wextra"
  "-Wpedantic"
  "-Wno-unused-local-typedefs")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND KAMINPAR_WARNING_FLAGS
    "-Wextra-semi"
    "-fcolor-diagnostics"
    "-Wdeprecated")
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(APPEND KAMINPAR_WARNING_FLAGS
    "-Wsuggest-override"
    "-fdiagnostics-color=always"
    "-Wcast-qual"
    "-Winit-self"
    "-Woverloaded-virtual"
    "-Wredundant-decls"
    "-Wno-psabi")
endif()

# Build experimental features that increase compile times
if(KAMINPAR_BUILD_EXPERIMENTAL_FEATURES)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_EXPERIMENTAL")
endif()

# Always enable Debug symbols (including in Release mode)
if(KAMINPAR_BUILD_WITH_DEBUG_SYMBOLS)
  add_compile_options(-g)
endif()

# Set compile flags
set(CMAKE_REQUIRED_FLAGS -Werror) # otherwise the check fails for Apple Clang

check_cxx_compiler_flag(-msse4.1 COMPILER_SUPPORTS_MSSE41)
if(COMPILER_SUPPORTS_MSSE41)
  add_compile_options(-msse4.1)
endif()

check_cxx_compiler_flag(-mcx16 COMPILER_SUPPORTS_MCX16)
if(COMPILER_SUPPORTS_MCX16)
  add_compile_options(-mcx16)
endif()

if(KAMINPAR_BUILD_WITH_MTUNE_NATIVE)
  add_compile_options(-mtune=native -march=native)
endif()

if(KAMINPAR_BUILD_WITH_ASAN)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

if(KAMINPAR_BUILD_WITH_UBSAN)
  add_compile_options(-fsanitize=undefined)
  add_link_options(-fsanitize=undefined)
endif()

if(KAMINPAR_BUILD_WITH_PG)
  add_compile_options(-pg)
endif()

# Pass CMake options to code
if(KAMINPAR_ENABLE_STATISTICS)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_STATISTICS")
  message(STATUS "Statistics: enabled")
else()
  message(STATUS "Statistics: disabled")
endif()

if(KAMINPAR_ENABLE_HEAP_PROFILING)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_HEAP_PROFILING")
  message(STATUS "Heap Profiling: enabled")
else()
  message(STATUS "Heap Profiling: disabled")
endif()

if(KAMINPAR_ENABLE_PAGE_PROFILING)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_PAGE_PROFILING")
  message(STATUS "Page Profiling: enabled")
else()
  message(STATUS "Page Profiling: disabled")
endif()

if(KAMINPAR_ENABLE_TIMERS)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_TIMERS")
  message(STATUS "Timers: enabled")
else()
  message(STATUS "Timers: disabled")
endif()

if(KAMINPAR_ENABLE_TIMER_BARRIERS)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_TIMER_BARRIERS")
  message(STATUS "Timer barriers: enabled")
else()
  message(STATUS "Timer barriers: disabled")
endif()

if(KAMINPAR_ENABLE_TBB_MALLOC)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_TBB_MALLOC")
  message(STATUS "TBB malloc: enabled")
else()
  message(STATUS "TBB malloc: disabled")
endif()

if(KAMINPAR_ENABLE_THP)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_ENABLE_THP")
  message(STATUS "Huge pages: enabled")
else()
  message(STATUS "Huge pages: disabled")
endif()

message(STATUS "Graph compression summary:")

if(KAMINPAR_COMPRESSION_HIGH_DEGREE_ENCODING)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_COMPRESSION_HIGH_DEGREE_ENCODING")
  message("  High-degree encoding: enabled")
else()
  message("  High-degree encoding: disabled")
endif()

if(KAMINPAR_COMPRESSION_INTERVAL_ENCODING)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_COMPRESSION_INTERVAL_ENCODING")
  message("  Interval encoding: enabled")
else()
  message("  Interval encoding: disabled")
endif()

if(KAMINPAR_COMPRESSION_RUN_LENGTH_ENCODING)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_COMPRESSION_RUN_LENGTH_ENCODING")
  message("  Run-length encoding: enabled")
else()
  message("  Run-length encoding: disabled")
endif()

if(KAMINPAR_COMPRESSION_STREAMVBYTE_ENCODING)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_COMPRESSION_STREAMVBYTE_ENCODING")
  message("  StreamVByte encoding: enabled")
else()
  message("  StreamVByte encoding: disabled")
endif()

if(KAMINPAR_COMPRESSION_FAST_DECODING)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_COMPRESSION_FAST_DECODING")
  add_compile_options(-mbmi2)
  message("  Fast decoding: enabled")
else()
  message("  Fast decoding: disabled")
endif()

if(KAMINPAR_64BIT_NODE_IDS OR KAMINPAR_64BIT_IDS)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_NODE_IDS")
  set(KAMINPAR_SHM_NODE_ID_STR "std::uint64_t")
else()
  set(KAMINPAR_SHM_NODE_ID_STR "std::uint32_t")
endif()

if(KAMINPAR_64BIT_EDGE_IDS OR KAMINPAR_64BIT_IDS)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_EDGE_IDS")
  set(KAMINPAR_SHM_EDGE_ID_STR "std::uint64_t")
else()
  set(KAMINPAR_SHM_EDGE_ID_STR "std::uint32_t")
endif()

if(KAMINPAR_64BIT_WEIGHTS)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_WEIGHTS")
  set(KAMINPAR_SHM_WEIGHT_STR "std::int64_t")
else()
  set(KAMINPAR_SHM_WEIGHT_STR "std::int32_t")
endif()

if(KAMINPAR_64BIT_LOCAL_WEIGHTS)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_LOCAL_WEIGHTS")
  set(KAMINPAR_DIST_WEIGHT_STR "std::int64_t")
else()
  set(KAMINPAR_DIST_WEIGHT_STR "std::int32_t")
endif()

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
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

include(FindGit)
include(FetchContent)

find_library(NUMA_LIB numa)

if(KAMINPAR_BUILD_WITH_CCACHE)
  find_program(CCACHE_PROGRAM ccache)

  if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()
endif()

# We use Google Sparsehash for hash tables
if(KAMINPAR_BUILD_WITH_SPARSEHASH)
  find_package(Sparsehash REQUIRED)
  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_SPARSEHASH_FOUND")
endif()

# We use Intel' TBB as a parallelization library
if(NOT KAMINPAR_DOWNLOAD_TBB)
  find_package(TBB QUIET)
endif()

if(TBB_FOUND)
  message(STATUS "Found TBB version ${TBB_VERSION} on the system")
else()
  message(STATUS "Using FetchContent to get TBB...")
  set(KAMINPAR_DOWNLOAD_TBB ON)

  FetchContent_Declare(TBB
    GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
    GIT_TAG 06ce6212da6710f4bb2d20a1904b018aa44069bf # v2022.2.0
    EXCLUDE_FROM_ALL
    SYSTEM)
  set(TBB_TEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(TBB)
endif()

# We use KAssert for assertions
if(KAMINPAR_BUILD_WITH_KASSERT)
  if(NOT KAMINPAR_DOWNLOAD_KASSERT)
    find_package(kassert QUIET)
  endif()

  if(kassert_FOUND)
    message(STATUS "Found kassert version ${kassert_VERSION} on the system")
  else()
    message(STATUS "Using FetchContent to get kassert...")
    set(KAMINPAR_DOWNLOAD_KASSERT ON)

    FetchContent_Declare(
      kassert
      GIT_REPOSITORY https://github.com/kamping-site/kassert.git
      GIT_TAG 988b7d54b79ae6634f2fcc53a0314fb1cf2c6a23
      EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(kassert)
  endif()

  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_KASSERT_FOUND")
endif()

# We optionally use Mt-KaHyPar as an initial partitioner and refiner
if(KAMINPAR_BUILD_WITH_MTKAHYPAR)
  if(NOT KAMINPAR_DOWNLOAD_MTKAHYPAR)
    find_package(MtKaHyPar QUIET)
  endif()

  if(MtKaHyPar_FOUND)
    message(STATUS "Found Mt-KaHyPar version ${MtKaHyPar_VERSION} on the system")
  else()
    message(STATUS "Using FetchContent to get Mt-KaHyPar...")
    set(KAMINPAR_DOWNLOAD_MTKAHYPAR ON)

    FetchContent_Declare(
      MtKaHyPar
      GIT_REPOSITORY https://github.com/kahypar/mt-kahypar.git
      GIT_TAG a0236788100facfce4ffae29263c4cf26c3cafcd # v1.5.2
      EXCLUDE_FROM_ALL)

    set(KAHYPAR_ENABLE_ARCH_COMPILE_OPTIMIZATIONS ${KAMINPAR_BUILD_WITH_MTUNE_NATIVE} CACHE BOOL "" FORCE)
    if(KAMINPAR_64BIT_IDS OR KAMINPAR_64BIT_NODE_IDS OR KAMINPAR_64BIT_EDGE_IDS)
      set(KAHYPAR_USE_64_BIT_IDS ON CACHE BOOL "" FORCE)
    endif()

    FetchContent_MakeAvailable(MtKaHyPar)
  endif()

  list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_MTKAHYPAR_FOUND")
endif()

if(KAMINPAR_BUILD_DISTRIBUTED)
  # We use MPI for message passing
  set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
  find_package(MPI QUIET)
  if(NOT MPI_FOUND)
    message(WARNING "MPI not available: cannot build the distributed partitioner")
    set(KAMINPAR_BUILD_DISTRIBUTED OFF)
  endif()

  # We use Growt for dynamically growing concurrent hash tables
  add_subdirectory(external/growt EXCLUDE_FROM_ALL)
  add_library(growt INTERFACE)
  target_include_directories(growt SYSTEM INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/growt>)

  # We optionally use Backward for pretty-printing stack traces
  if(KAMINPAR_BUILD_WITH_BACKWARD)
    FetchContent_Declare(
      Backward
      GIT_REPOSITORY https://github.com/kamping-site/bakward-mpi.git
      GIT_TAG 89de1132cdccb60aa4994d00396cc30d47402f95
      EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(Backward)
  endif()
endif()

# We use KaGen for distributed graph IO
if((KAMINPAR_BUILD_DISTRIBUTED AND KAMINPAR_BUILD_APPS) OR KAMINPAR_BUILD_BENCHMARKS)
  if(NOT KAMINPAR_DOWNLOAD_KAGEN)
    find_package(KaGen QUIET)
  endif()

  if(KaGen_FOUND)
    message(STATUS "Found KaGen version ${KaGen_VERSION} on the system")
  else()
    message(STATUS "Using FetchContent to get KaGen...")
    set(KAMINPAR_DOWNLOAD_KAGEN ON)

    FetchContent_Declare(
      KaGen
      GIT_REPOSITORY https://github.com/KarlsruheGraphGeneration/KaGen.git
      GIT_TAG 70386f48e513051656f020360c482ce6bff9a24f
      PATCH_COMMAND ${GIT_EXECUTABLE} apply ${CMAKE_CURRENT_SOURCE_DIR}/scripts/KaGen.patch
      EXCLUDE_FROM_ALL)
    set(KAGEN_NODEPS ON CACHE BOOL "" FORCE)
    set(KAGEN_BUILD_APPS OFF CACHE BOOL "" FORCE)
    set(KAGEN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(KAGEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(INSTALL_KAGEN OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(KaGen)
  endif()
endif()

# We use GoogleTest for unit tests
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR KAMINPAR_BUILD_TESTS) AND BUILD_TESTING)
  if(NOT KAMINPAR_DOWNLOAD_GTEST)
    find_package(GTest QUIET)
  endif()

  if(GTest_FOUND)
    message(STATUS "Found GoogleTest version ${GTest_VERSION} on the system")
  else()
    message(STATUS "Using FetchContent to get GoogleTest...")

    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG 52eb8108c5bdec04579160ae17225d66034bd723
      EXCLUDE_FROM_ALL)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
endif()

################################################################################
## Add targets in subdirectories                                              ##
################################################################################

# Start include paths on project root
include_directories(${PROJECT_SOURCE_DIR})

# Add core library targets
add_subdirectory(kaminpar-common) # Common utilities shared across the project
add_subdirectory(kaminpar-shm) # Shared-memory components

if(KAMINPAR_BUILD_DISTRIBUTED)
  add_subdirectory(kaminpar-mpi) # MPI-related utilities
  add_subdirectory(kaminpar-dist) # Distributed components
endif()

add_subdirectory(kaminpar-io) # IO components

# Create an interface library that groups all core components and ensures the full
# project include path is available to dependent targets
add_library(KaMinParFull INTERFACE)
add_library(KaMinPar::KaMinParFull ALIAS KaMinParFull)

target_include_directories(KaMinParFull INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(KaMinParFull INTERFACE KaMinParIO KaMinPar)
if(KAMINPAR_BUILD_DISTRIBUTED)
  target_link_libraries(KaMinParFull INTERFACE dKaMinParIO dKaMinPar)
endif()

# Tests are built either when running as the main project or when explicitly enabled
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR KAMINPAR_BUILD_TESTS) AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(KAMINPAR_BUILD_APPS)
  add_subdirectory(kaminpar-cli)
  add_subdirectory(apps)
endif()

if(KAMINPAR_BUILD_EXAMPLES)
  add_subdirectory(examples/kaminpar)
  add_subdirectory(examples/bindings-c)
endif()

################################################################################
## Install targets                                                            ##
################################################################################
if(NOT INSTALL_KAMINPAR)
  return()
endif()

# If the installation prefix is not set by the user for Unix builds,
# assign a default value to ensure GNUInstallDirs selects the correct paths.
if(UNIX AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/usr/" CACHE STRING "" FORCE)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(KAMINPAR_INSTALL_INCLUDE_DIRS include/kaminpar-io/ include/kaminpar-shm/)
if(KAMINPAR_BUILD_DISTRIBUTED)
  list(APPEND KAMINPAR_INSTALL_INCLUDE_DIRS include/kaminpar-dist/)
endif()

install(
    DIRECTORY ${KAMINPAR_INSTALL_INCLUDE_DIRS}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT KaMinPar_Development
)

set(KAMINPAR_INSTALL_TARGETS KaMinParCommon KaMinParIO KaMinPar)

if(KAMINPAR_BUILD_WITH_KASSERT AND KAMINPAR_DOWNLOAD_KASSERT)
  list(APPEND KAMINPAR_INSTALL_TARGETS kassert_base kassert)
endif()

if(KAMINPAR_BUILD_WITH_MTKAHYPAR AND KAMINPAR_DOWNLOAD_MTKAHYPAR)
  list(APPEND KAMINPAR_INSTALL_TARGETS mtkahypar)
endif()

if(KAMINPAR_BUILD_APPS)
  list(APPEND KAMINPAR_INSTALL_TARGETS KaMinParCLI KaMinParApp)
endif()

if(KAMINPAR_BUILD_DISTRIBUTED)
  list(APPEND KAMINPAR_INSTALL_TARGETS growt KaMinParMPI dKaMinParIO dKaMinPar)

  if (KAMINPAR_DOWNLOAD_KAGEN)
    list(APPEND KAMINPAR_INSTALL_TARGETS kagen morton)
  endif ()

  if(KAMINPAR_BUILD_APPS)
    list(APPEND KAMINPAR_INSTALL_TARGETS dKaMinParCLI dKaMinParApp)
  endif()
endif()

install(
  TARGETS ${KAMINPAR_INSTALL_TARGETS}
  EXPORT KaMinParTargets
  RUNTIME COMPONENT KaMinPar_Runtime
  LIBRARY COMPONENT KaMinPar_Runtime NAMELINK_COMPONENT KaMinPar_Development
  ARCHIVE COMPONENT KaMinPar_Development
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Allow package maintainers to freely override the path for the configs
set(
  KAMINPAR_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/KaMinPar"
  CACHE STRING "CMake package config location relative to the install prefix"
)
set_property(CACHE KAMINPAR_INSTALL_CMAKEDIR PROPERTY TYPE PATH)
mark_as_advanced(KAMINPAR_INSTALL_CMAKEDIR)

install(
  FILES ${PROJECT_SOURCE_DIR}/cmake/modules/FindSparsehash.cmake
  DESTINATION ${KAMINPAR_INSTALL_CMAKEDIR}
  COMPONENT KaMinPar_Development
)

configure_file(cmake/KaMinParConfig.cmake.in KaMinParConfig.cmake @ONLY)
install(
    FILES ${PROJECT_BINARY_DIR}/KaMinParConfig.cmake
    DESTINATION ${KAMINPAR_INSTALL_CMAKEDIR}
    COMPONENT KaMinPar_Development
)

write_basic_package_version_file(
  "KaMinParConfigVersion.cmake"
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(
  FILES ${PROJECT_BINARY_DIR}/KaMinParConfigVersion.cmake
  DESTINATION ${KAMINPAR_INSTALL_CMAKEDIR}
  COMPONENT KaMinPar_Development
)

install(
  EXPORT KaMinParTargets
  NAMESPACE KaMinPar::
  DESTINATION ${KAMINPAR_INSTALL_CMAKEDIR}
  COMPONENT KaMinPar_Development
)

################################################################################
## Package targets                                                            ##
################################################################################
set(CPACK_THREADS 0) # Given 0 CPack will try to use all available CPU cores.
set(CPACK_VERBATIM_VARIABLES ON)

set(CPACK_PACKAGE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/packages")
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "")

set(CPACK_PACKAGE_VENDOR ${PROJECT_VENDOR})
set(CPACK_PACKAGE_CONTACT "${PROJECT_VENDOR} <${PROJECT_CONTACT}>")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.MD")

set(CPACK_STRIP_FILES ON)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
set(
  CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
  OWNER_READ OWNER_WRITE OWNER_EXECUTE
  GROUP_READ GROUP_EXECUTE
  WORLD_READ WORLD_EXECUTE
)

set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_RELEASE 1)
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.34), libgcc-s1 (>= 3.0), libstdc++6 (>= 10.2), libtbb-dev (>= 2018~U6-4), libsparsehash-dev (>= 2.0.2-1)")
if(NUMA_LIB)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libnuma-dev (>= 2.0.12-1)")
endif()
if(KAMINPAR_BUILD_DISTRIBUTED)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libopenmpi-dev (>= 3.1.3-11)")
endif()

set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_RELEASE 1)
set(CPACK_RPM_PACKAGE_AUTOREQ OFF)
set(CPACK_RPM_PACKAGE_REQUIRES "tbb-devel, sparsehash-devel")
if(NUMA_LIB)
  set(CPACK_RPM_PACKAGE_REQUIRES "${CPACK_RPM_PACKAGE_REQUIRES}, numactl-devel")
endif()
if(KAMINPAR_BUILD_DISTRIBUTED)
  set(CPACK_RPM_PACKAGE_REQUIRES "${CPACK_RPM_PACKAGE_REQUIRES}, openmpi-devel")
endif()

include(CPack)
