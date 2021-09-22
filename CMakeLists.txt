cmake_minimum_required(VERSION 3.16)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

project(KaMinPar CXX C)
set(PROJECT_VENDOR "Daniel Seemaier")
set(PROJECT_CONTACT "daniel.seemaier@kit.edu")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# Default to Release mode
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()

option(KAMINPAR_ENABLE_HEAVY_ASSERTIONS "Enable assertions that have high impact on performance." OFF)
option(KAMINPAR_ENABLE_LIGHT_ASSERTIONS "Enable very lightweight assertions." ON)
option(KAMINPAR_ENABLE_TIMERS "Enable fine-grained timers. With this option disabled, we will still report the total partitioning time." ON)
option(KAMINPAR_ENABLE_DEBUG_FEATURES "Enable features that might slow down execution but are helpful for debug purposes (e.g., naming graphs)." OFF)
option(KAMINPAR_ENABLE_STATISTICS "Generate and output much more detailed statistics." OFF)
option(KAMINPAR_64BIT_EDGE_IDS "Use 64 bits for edge ids." OFF)
option(KAMINPAR_USE_BACKWARD_CPP "Use backward-cpp for better stacktraces." OFF)
option(KAMINPAR_BUILD_TESTS "If enabled, build unit tests." OFF)

include_directories(${PROJECT_SOURCE_DIR})

# ensure that gmock is built before tests are executed
if (KAMINPAR_BUILD_TESTS)
    add_subdirectory(external_tools/googletest EXCLUDE_FROM_ALL)
    include_directories(SYSTEM ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
    include_directories(SYSTEM ${gtest_SOURCE_DIR}/../googletest/include ${gtest_SOURCE_DIR}/../googletest)
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wextra ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverloaded-virtual")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1 -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -mtune=native -march=native")

set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")

# if we want to debug in Release mode
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fsanitize=address -fno-omit-frame-pointer")
#set(CMAKE_LINKER_FLAGS_RELEASE "${CMAKE_LINKER_FLAGS_RELEASE} -fsanitize=address")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_LINKER_FLAGS_DEBUG: ${CMAKE_LINKER_FLAGS_DEBUG}")

if (KAMINPAR_ENABLE_HEAVY_ASSERTIONS)
    add_definitions(-DKAMINPAR_ENABLE_HEAVY_ASSERTIONS)
    message(STATUS "Building with heavy assertions enabled.")
endif ()

if (KAMINPAR_ENABLE_LIGHT_ASSERTIONS)
    add_definitions(-DKAMINPAR_ENABLE_LIGHT_ASSERTIONS)
    message(STATUS "Building with lightweight assertions enabled.")
endif ()

if (KAMINPAR_ENABLE_DEBUG_FEATURES)
    add_definitions(-DKAMINPAR_ENABLE_DEBUG_FEATURES)
    message(STATUS "Building with global debug features enabled.")
endif ()

if (KAMINPAR_ENABLE_TIMERS)
    add_definitions(-DKAMINPAR_ENABLE_TIMERS)
    message(STATUS "Building with fine-grained timers enabled.")
endif ()

if (KAMINPAR_ENABLE_STATISTICS)
    add_definitions(-DKAMINPAR_ENABLE_STATISTICS)
endif ()

if (KAMINPAR_64BIT_EDGE_IDS)
    add_definitions(-DKAMINPAR_64BIT_EDGE_IDS)
endif ()

if (KAMINPAR_BUILD_TESTS)
    add_definitions(-DKAMINPAR_BUILD_TESTS)
endif ()

include(gmock)

# Growt
add_library(growt INTERFACE)
target_include_directories(growt INTERFACE external_tools/growt/)

# Backward-cpp for stack traces in MPI code
if (KAMINPAR_USE_BACKWARD_CPP)
    set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
    add_subdirectory(external_tools/backward-cpp)
endif (KAMINPAR_USE_BACKWARD_CPP)

add_subdirectory(kaminpar)
add_subdirectory(dkaminpar)
add_subdirectory(apps)
add_subdirectory(library)

if (KAMINPAR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
#    add_subdirectory(dtests)
endif ()