FetchContent_MakeAvailable(googletest)
set_property(DIRECTORY "${googletest_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL YES)

enable_testing()

include(GoogleTest)
include(cmake/KaTestrophe.cmake)

function(kaminpar_add_shm_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE 
        GTest::gtest_main 
        GTest::gmock_main
        common_base 
        shm_partitioner_base)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    gtest_discover_tests(${target})
endfunction()

function(kaminpar_add_common_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE 
        GTest::gtest_main
        GTest::gmock_main
        common_base)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    gtest_discover_tests(${target})
endfunction()

function(kaminpar_add_dist_test KAMINPAR_TARGET_NAME)
    cmake_parse_arguments(
            "KAMINPAR"
            ""
            ""
            "FILES;CORES"
            ${ARGN}
    )
    katestrophe_add_test_executable(${KAMINPAR_TARGET_NAME} FILES ${KAMINPAR_FILES})
    target_link_libraries(${KAMINPAR_TARGET_NAME} PRIVATE 
        mpi-gtest-main
        common_base 
        shm_partitioner_base 
        dist_partitioner_base)
    target_include_directories(${KAMINPAR_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    katestrophe_add_mpi_test(${KAMINPAR_TARGET_NAME} CORES ${KAMINPAR_CORES})
endfunction()

# Copy test instances to binary directory
file(COPY test_instances DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Common -> Datastructures
kaminpar_add_common_test(test_common_preallocated_vector common/datastructures/preallocated_vector_test.cc)
kaminpar_add_common_test(test_common_binary_heap common/datastructures/binary_heap_test.cc)
kaminpar_add_common_test(test_common_fast_reset_array common/datastructures/fast_reset_array_test.cc)
kaminpar_add_common_test(test_common_marker common/datastructures/marker_test.cc)
kaminpar_add_common_test(test_common_static_array common/datastructures/static_array_test.cc)

# Common 
kaminpar_add_common_test(test_common_math common/math_test.cc)
kaminpar_add_common_test(test_common_string common/strutils_test.cc)
kaminpar_add_common_test(test_common_parallel_algorithm common/parallel/algorithm_test.cc)

# dKaMinPar -> Graph Utils
kaminpar_add_dist_test(test_dist_graph_extraction
    FILES dkaminpar/graphutils/graph_extraction_test.cc
    CORES 1 2 4)

kaminpar_add_dist_test(test_dist_graph_rearrangement
    FILES dkaminpar/graphutils/graph_rearrangement_test.cc
    CORES 1 2 3 4)

kaminpar_add_dist_test(test_dist_independent_set
    FILES dkaminpar/graphutils/independent_set_test.cc
    CORES 1 2 3 4)

kaminpar_add_dist_test(test_dist_bfs_extractor
    FILES dkaminpar/graphutils/bfs_extractor_test.cc
    CORES 1 2 4)

kaminpar_add_dist_test(test_dist_global_move_conflict_resolver
    FILES dkaminpar/refinement/move_conflict_resolver_test.cc
    CORES 1 2 3 4)

kaminpar_add_dist_test(test_dist_sparse_alltoall
    FILES dkaminpar/sparse_alltoall_test.cc
    CORES 1 2 3 4 11)

kaminpar_add_dist_test(test_dist_graph_topology
    FILES dkaminpar/mpi/grid_topology_test.cc
    CORES 1)

kaminpar_add_dist_test(test_dist_graph_allgather
    FILES dkaminpar/graphutils/allgather_graph_test.cc
    CORES 1 2 4)

# dKaMinPar -> Algorithms 
kaminpar_add_dist_test(test_dist_greedy_node_coloring
    FILES dkaminpar/algorithms/greedy_node_coloring_test.cc
    CORES 1)

# KaMinPar 
kaminpar_add_shm_test(test_shm_io kaminpar/io_test.cc)
kaminpar_add_shm_test(test_shm_metrics kaminpar/metrics_test.cc)
kaminpar_add_shm_test(test_shm_graph kaminpar/graph_test.cc)
kaminpar_add_shm_test(test_subgraph_extraction kaminpar/subgraph_extraction_test.cc)
kaminpar_add_shm_test(test_graph_utils kaminpar/graph_utils_test.cc)
