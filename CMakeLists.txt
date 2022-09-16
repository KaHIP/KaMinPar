cmake_minimum_required(VERSION 3.16)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

project(KaMinPar
        DESCRIPTION "Shared-memory and distributed graph partitioner for Large K Partitioning"
        LANGUAGES CXX)

set(PROJECT_VENDOR "Daniel Seemaier")
set(PROJECT_CONTACT "daniel.seemaier@kit.edu")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# Default to Release mode
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()

option(KAMINPAR_ENABLE_STATISTICS "Generate and output detailed statistics." ON)
option(KAMINPAR_ENABLE_BACKWARD_CPP "Build with backward-cpp for better stack traces (requires the backward-cpp submodule)." ON)
option(KAMINPAR_ENABLE_GRAPHGEN "Build with support for in-memory graph generation (requires the KaGen submodule)." ON)
option(KAMINPAR_ENABLE_SANITIZER "Enable address sanitizer." OFF)
option(KAMINPAR_ENABLE_MTUNE_NATIVE "Build with -mtune=native." ON)

option(KAMINPAR_ENABLE_MTKAHYPAR "Enable Mt-KaHyPar as initial partitioner for the distributed partitioner." OFF)

option(KAMINPAR_BUILD_TESTS "Build unit tests (requires the GoogleTest submodule)." ON)
option(KAMINPAR_BUILD_DISTRIBUTED "Enable or disable the distributed partitioner (requires the growt submodule)." ON)

option(KAMINPAR_64BIT_IDS "Use 64 bits for node and edge IDs." OFF)
option(KAMINPAR_64BIT_EDGE_IDS "Use 64 bits for edge IDs." OFF)
option(KAMINPAR_64BIT_NODE_IDS "Use 64 bits for node IDs." OFF)
option(KAMINPAR_64BIT_WEIGHTS "Use 64 bit for node and edge weights." OFF)

set(KAMINPAR_ASSERTION_LEVEL "light" CACHE STRING "Assertion level.")
set_property(CACHE KAMINPAR_ASSERTION_LEVEL PROPERTY STRINGS none light normal heavy)
message(STATUS "Assertion level: ${KAMINPAR_ASSERTION_LEVEL}")

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverloaded-virtual")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Growt requires -mcx16 (x86 only)
if (KAMINPAR_BUILD_DISTRIBUTED)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-mcx16 COMPILER_SUPPORTS_MCX16)
    if (COMPILER_SUPPORTS_MCX16) 
        add_compile_options(-mcx16)
    else ()
        message(WARNING "Cannot build distributed partitioner: compiler does not support the -mcx16 flag.")
    endif()
endif ()

# Set compile flags 
if (KAMINPAR_ENABLE_MTUNE_NATIVE) 
    add_compile_options(-mtune=native -march=native)
endif ()

if (KAMINPAR_ENABLE_SANITIZER) 
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

# Pass CMake options to code
if (KAMINPAR_ENABLE_STATISTICS)
    add_definitions(-DKAMINPAR_ENABLE_STATISTICS)
    message(STATUS "Enable statistics.")
endif ()

if (KAMINPAR_64BIT_NODE_IDS OR KAMINPAR_64BIT_IDS)
    add_definitions(-DKAMINPAR_64BIT_NODE_IDS)
    message(STATUS "Use 64 bit node IDs.")
endif ()

if (KAMINPAR_64BIT_EDGE_IDS OR KAMINPAR_64BIT_IDS)
    add_definitions(-DKAMINPAR_64BIT_EDGE_IDS)
    message(STATUS "Use 64 bit edge IDs.")
endif ()

if (KAMINPAR_64BIT_WEIGHTS)
    add_definitions(-DKAMINPAR_64BIT_WEIGHTS)
    message(STATUS "Use 64 bit node and edge weights.")
endif ()

if (KAMINPAR_BUILD_TESTS)
    add_definitions(-DKAMINPAR_BUILD_TESTS)
    add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")
    include(gmock)

    message(STATUS "Build unit tests.")
endif ()

# Backward-cpp for stack traces in MPI code
if (KAMINPAR_ENABLE_BACKWARD_CPP)
    set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
    add_subdirectory(extern/backward-cpp)
    add_definitions(-DKAMINPAR_ENABLE_BACKWARD_CPP)
    message(STATUS "Enable backward-cpp.")
endif ()

# Use ccache is available 
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    message(STATUS "Use ccache.")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif ()

# Look for MPI 
if (KAMINPAR_BUILD_DISTRIBUTED AND COMPILER_SUPPORTS_MCX16) 
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
    find_package(MPI)

    if (NOT MPI_FOUND) 
        message(WARNING "Cannout build distributed partitioner: could not find MPI")
    else () # Include growt 
        add_library(growt INTERFACE)
        target_include_directories(growt INTERFACE extern/growt/)
    endif ()
endif ()

# Include kassert 
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
add_subdirectory(extern/kassert)

# Start include paths on project root
include_directories(${PROJECT_SOURCE_DIR})

# Shared memory components
add_subdirectory(common)
add_subdirectory(kaminpar)
add_subdirectory(library)

# Distributed components
if (KAMINPAR_BUILD_DISTRIBUTED AND COMPILER_SUPPORTS_MCX16 AND MPI_FOUND)
    if (KAMINPAR_ENABLE_GRAPHGEN) # KaGen
        set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
        set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
        add_subdirectory(extern/KaGen/)
        add_definitions(-DKAMINPAR_ENABLE_GRAPHGEN)
    endif ()

    add_subdirectory(dkaminpar)
endif ()

# Binaries
add_subdirectory(apps)

# Unit tests
if (KAMINPAR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

