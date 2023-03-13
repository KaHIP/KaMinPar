cmake_minimum_required(VERSION 3.25)
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
option(KAMINPAR_TESTS "Build unit tests" ON)
option(KAMINPAR_DISTRIBUTED "Enable or disable the distributed partitioner (requires the growt submodule)." ON)
option(KAMINPAR_MTKAHYPAR "Enable Mt-KaHyPar as initial partitioner for the distributed partitioner." OFF)
option(KAMINPAR_PYTHON_MODULE "Build the Python binding." OFF)
option(KAMINPAR_METIS_INTERFACE "Build the METIS interface." OFF)

# Control how to build
option(KAMINPAR_STATISTICS "Generate and output detailed statistics." ON)
option(KAMINPAR_SANITIZER "Enable address sanitizer." OFF)
option(KAMINPAR_MTUNE_NATIVE "Build with -mtune=native." ON)
option(KAMINPAR_CCACHE "Use ccache to build." ON)

# Control data type sizes
option(KAMINPAR_64BIT_IDS "Use 64 bits for node and edge IDs." OFF)
option(KAMINPAR_64BIT_EDGE_IDS "Use 64 bits for edge IDs." OFF)
option(KAMINPAR_64BIT_NODE_IDS "Use 64 bits for node IDs." OFF)
option(KAMINPAR_64BIT_WEIGHTS "Use 64 bit for node and edge weights." OFF)
option(KAMINPAR_64BIT_LOCAL_WEIGHTS "Use 64 bit for local node and edge weights." OFF)

################################################################################
## Declare dependencies                                                       ##
################################################################################

FetchContent_Declare(
    googletest
    SYSTEM
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1)

FetchContent_Declare(
    kassert
    GIT_REPOSITORY https://github.com/kamping-site/kassert.git
    GIT_TAG main)

FetchContent_Declare(
    growt
    SYSTEM
    GIT_REPOSITORY https://github.com/DanielSeemaier/growt.git
    GIT_TAG kaminpar)

set(KAMINPAR_ASSERTION_LEVEL "light" CACHE STRING "Assertion level.")
set_property(CACHE KAMINPAR_ASSERTION_LEVEL PROPERTY STRINGS none light normal heavy)
message(STATUS "KAssertion level: ${KAMINPAR_ASSERTION_LEVEL}")

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Og")

# Set compile flags 
if (KAMINPAR_MTUNE_NATIVE) 
    add_compile_options(-mtune=native -march=native)
endif ()

if (KAMINPAR_SANITIZER) 
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

set(KAMINPAR_DEFINITIONS "")

# Pass CMake options to code
if (KAMINPAR_STATISTICS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_STATISTICS")
    message(STATUS "Statistics: enabled")
else ()
    message(STATIS "Statistics: disabled")
endif ()

if (KAMINPAR_64BIT_NODE_IDS OR KAMINPAR_64BIT_IDS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_NODE_IDS")
    message(STATUS "Node ID size: 64 bit")
else ()
    message(STATUS "Node ID size: 32 bit")
endif ()

if (KAMINPAR_64BIT_EDGE_IDS OR KAMINPAR_64BIT_IDS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_EDGE_IDS")
    message(STATUS "Edge ID size: 64 bit")
else ()
    message(STATUS "Edge ID size: 32 bit")
endif ()

if (KAMINPAR_64BIT_WEIGHTS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_WEIGHTS")
    message(STATUS "Node and edge weight size: 64 bit")
else ()
    message(STATUS "Node and edge weight size: 32 bit")
endif ()

if (KAMINPAR_64BIT_LOCAL_WEIGHTS)
    list(APPEND KAMINPAR_DEFINITIONS "-DKAMINPAR_64BIT_LOCAL_WEIGHTS")
    message(STATUS "Local node and edge weight size: 64 bit (distributed only)")
else ()
    message(STATUS "Local node and edge weight size: 32 bit (distributed only)")
endif ()

################################################################################
## Search and fetch dependencies                                              ##
################################################################################

if (KAMINPAR_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif ()
endif ()

if (KAMINPAR_DISTRIBUTED)
    # MPI
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
    find_package(MPI)
    if (NOT MPI_FOUND) 
        message(WARNING "MPI not available: cannot build the distributed partitioner")
        set(KAMINPAR_DISTRIBUTED OFF)
    endif ()

    # Growt (needs -mcx16, i.e., does not work on ARM)
    message("################################################################################")
    message("## growt                                                                      ##")
    message("################################################################################")

    check_cxx_compiler_flag(-mcx16 COMPILER_SUPPORTS_MCX16)
    if (COMPILER_SUPPORTS_MCX16) 
        add_compile_options(-mcx16)
    else ()
        message(WARNING "-mcx16 flag not supported by the compiler: cannot build the distributed partitioner")
        set(KAMINPAR_DISTRIBUTED OFF)
    endif()

    FetchContent_MakeAvailable(growt)
    set_property(DIRECTORY "${growt_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL YES)

    # growt does not add a library target; we have to do that ourself
    add_library(growt INTERFACE)
    target_include_directories(growt SYSTEM INTERFACE "${growt_SOURCE_DIR}")

    # Google Sparsehash 
    find_package(Sparsehash REQUIRED)
endif ()

message("################################################################################")
message("## KAssert                                                                    ##")
message("################################################################################")

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

FetchContent_MakeAvailable(kassert)
set_property(DIRECTORY "${kassert_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL YES)

message("--------------------------------------------------------------------------------")

################################################################################
## Add targets in subdirectories                                              ##
################################################################################

# Start include paths on project root
include_directories(${PROJECT_SOURCE_DIR})

# Shared memory components
add_subdirectory(common)
add_subdirectory(kaminpar)

# Distributed components
if (KAMINPAR_DISTRIBUTED)
    add_subdirectory(dkaminpar)
endif ()

# Binaries
add_subdirectory(apps)

# Unit tests
if (KAMINPAR_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

################################################################################

add_library(KaMinPar::KaMinPar ALIAS shm_partitioner_base)
if (KAMINPAR_DISTRIBUTED)
    add_library(KaMinPar::dKaMinPar ALIAS dist_partitioner_base)
endif ()
