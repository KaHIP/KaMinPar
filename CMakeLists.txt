cmake_minimum_required(VERSION 3.16)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

project(KaMinPar CXX C)
set(PROJECT_VENDOR "Daniel Seemaier")
set(PROJECT_CONTACT "daniel.seemaier@kit.edu")

set(CMAKE_CXX_STANDARD 20)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

option(KAMINPAR_LINK_TCMALLOC "Enable allocations using Google's tcmalloc." OFF)
option(KAMINPAR_ENABLE_HEAVY_ASSERTIONS "Enable assertions that have high impact on performance." OFF)
option(KAMINPAR_ENABLE_TIMERS "Enable fine-grained timers. With this option disabled, we will still report the total partitioning time." ON)
option(KAMINPAR_ENABLE_DEBUG_FEATURES "Enable features that might slow down execution but are helpful for debug purposes (e.g., naming graphs)." OFF)
option(KAMINPAR_ENABLE_STATISTICS "Generate and output much more detailed statistics." OFF)
option(KAMINPAR_64BIT_EDGE_IDS "Use 64 bits for edge ids." OFF)
option(KAMINPAR_64BIT_EDGE_WEIGHTS "Use 64 bits for edge weights." OFF)

include_directories(${PROJECT_SOURCE_DIR})

# ensure that gmock is built before tests are executed
add_subdirectory(external_tools/googletest EXCLUDE_FROM_ALL)
include_directories(SYSTEM ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(SYSTEM ${gtest_SOURCE_DIR}/../googletest/include ${gtest_SOURCE_DIR}/../googletest)

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

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")

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

if (KAMINPAR_ENABLE_DEBUG_FEATURES)
  add_definitions(-DKAMINPAR_ENABLE_DEBUG_FEATURES)
  message(STATUS "Building with global debug features enabled.")
endif ()

if (KAMINPAR_ENABLE_TIMERS)
  add_definitions(-DKAMINPAR_ENABLE_TIMERS)
  message(STATUS "Building with fine-grained timers enabled.")
endif ()

if (KAMINPAR_LINK_TCMALLOC)
  find_library(TCMALLOC_LIB tcmalloc)
  if (NOT TCMALLOC_LIB)
    message(STATUS "tcmalloc enabled, but unabled to find the library on this system.")
    set(TCMALLOC_LIB "")
  else ()
    message(STATUS "tcmalloc enabled and found at ${TCMALLOC_LIB}")
    add_definitions(-DKAMINPAR_LINK_TCMALLOC)
  endif ()
else ()
  set(TCMALLOC_LIB "")
  message(STATUS "tcmalloc disabled.")
endif ()

if (KAMINPAR_ENABLE_STATISTICS)
  add_definitions(-DKAMINPAR_ENABLE_STATISTICS)
endif ()

if (KAMINPAR_64BIT_EDGE_IDS)
  add_definitions(-DKAMINPAR_64BIT_EDGE_IDS)
endif ()
if (KAMINPAR_64BIT_EDGE_WEIGHTS)
  add_definitions(-DKAMINPAR_64BIT_EDGE_WEIGHTS)
endif ()

# Make commit hash available
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
add_definitions(-DGIT_COMMIT_HASH="${GIT_SHA1}")

# Make hostname available
include(GetHostname)
GetHostname(HOSTNAME)
add_definitions(-DHOSTNAME="${HOSTNAME}")

# Make list of modified files available
include(GetModifiedGitFiles)
GetModifiedGitFiles(MODIFIED_FILES)
add_definitions(-DMODIFIED_FILES="${MODIFIED_FILES}")

include(gmock)
add_subdirectory(kaminpar)
add_subdirectory(apps)
#add_subdirectory(tests)
add_subdirectory(library)
