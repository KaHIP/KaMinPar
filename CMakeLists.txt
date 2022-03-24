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

option(KAMINPAR_HEAVY_ASSERTIONS "Enable assertions that have high impact on performance." OFF)
option(KAMINPAR_LIGHT_ASSERTIONS "Enable very lightweight assertions." ON)

option(KAMINPAR_STATISTICS "Generate and output detailed statistics." OFF)
option(KAMINPAR_BACKWARD_CPP "Build with backward-cpp for better stack traces (requires the backward-cpp submodule)." ON)
option(KAMINPAR_TESTS "Build unit tests (requires the GoogleTest submodule)." ON)
option(KAMINPAR_GRAPHGEN "Build with support for in-memory graph generation (requires the KaGen submodule)." OFF)

option(KAMINPAR_64BIT_IDS "Use 64 bits for node and edge IDs." OFF)
option(KAMINPAR_64BIT_EDGE_IDS "Use 64 bits for edge IDs." OFF)
option(KAMINPAR_64BIT_NODE_IDS "Use 64 bits for node IDs." OFF)
option(KAMINPAR_64BIT_WEIGHTS "Use 64 bit for node and edge weights." OFF)

include_directories(${PROJECT_SOURCE_DIR})

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcx16") # growt

# Enable debug symbols and address sanitizer in Debug mode
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1 -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")

set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")

# Enable debug symbols and full optimization in Release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -mtune=native -march=native")
# (enable address sanitizer in Release mode:)
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fsanitize=address -fno-omit-frame-pointer")
#set(CMAKE_LINKER_FLAGS_RELEASE "${CMAKE_LINKER_FLAGS_RELEASE} -fsanitize=address")

# Pass CMake options to code
if (KAMINPAR_HEAVY_ASSERTIONS)
    add_definitions(-DKAMINPAR_ENABLE_HEAVY_ASSERTIONS)
    message(STATUS "Building with heavy assertions enabled.")
endif ()

if (KAMINPAR_LIGHT_ASSERTIONS)
    add_definitions(-DKAMINPAR_ENABLE_LIGHT_ASSERTIONS)
    message(STATUS "Building with light assertions enabled.")
endif ()

if (KAMINPAR_STATISTICS)
    add_definitions(-DKAMINPAR_ENABLE_STATISTICS)
    message(STATUS "Building with statistics enabled.")
endif ()

if (KAMINPAR_64BIT_EDGE_IDS OR KAMINPAR_64BIT_IDs)
    add_definitions(-DKAMINPAR_64BIT_EDGE_IDS)
    message(STATUS "Building with 64 bit edge IDs.")
endif ()

if (KAMINPAR_64BIT_NODE_IDS OR KAMINPAR_64BIT_IDs)
    add_definitions(-DKAMINPAR_64BIT_NODE_IDS)
    message(STATUS "Building with 64 bit node IDs.")
endif ()

if (KAMINPAR_TESTS)
    add_definitions(-DKAMINPAR_BUILD_TESTS)
    add_subdirectory("${PROJECT_SOURCE_DIR}/external_tools/googletest" "external_tools/googletest")
    include(gmock)

    message(STATUS "Building with unit tests.")
endif ()

if (KAMINPAR_64BIT_WEIGHTS)
    add_definitions(-DKAMINPAR_64BIT_WEIGHTS)
    message(STATUS "Building with 64 bit edge and node weights.")
endif ()

# Backward-cpp for stack traces in MPI code
if (KAMINPAR_BACKWARD_CPP)
    set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
    add_subdirectory(external_tools/backward-cpp)
    add_definitions(-DKAMINPAR_BACKWARD_CPP)
    message(STATUS "Building with backward-cpp.")
endif ()

# Look for MPI -- if this fails, we give up on the distributed components
set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
find_package(MPI)

# Only include Growt if we can build the distributed partitioner
if (MPI_FOUND)
    add_library(growt INTERFACE)
    target_include_directories(growt INTERFACE external_tools/growt/)
endif ()

# Shared memory components
add_subdirectory(kaminpar)
add_subdirectory(library)

# Distributed components
if (MPI_FOUND)
    if (KAMINPAR_GRAPHGEN) # KaGen
        set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
        set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
        add_subdirectory(external_tools/KaGen/)
        add_definitions(-DKAMINPAR_GRAPHGEN)
    endif ()

    #    add_subdirectory(dkaminpar)
endif ()

# Binaries
add_subdirectory(apps)

# Unit tests
if (KAMINPAR_TESTS)
    enable_testing()
    add_subdirectory(tests)

    if (MPI_FOUND)
	    #        add_subdirectory(dtests)
    endif ()
endif ()

