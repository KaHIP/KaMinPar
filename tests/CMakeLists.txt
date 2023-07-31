include(GoogleTest)

function(kaminpar_add_shm_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE 
        GTest::gtest_main 
        GTest::gmock_main
        common 
        kaminpar)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    gtest_discover_tests(${target})

    message(STATUS "Enabled shared-memory test: ${target}")
endfunction()

function(kaminpar_add_common_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE 
        GTest::gtest_main
        GTest::gmock_main
        common)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    gtest_discover_tests(${target})

    message(STATUS "Enabled common test: ${target}")
endfunction()

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

# KaMinPar 
kaminpar_add_shm_test(test_shm_metrics kaminpar/metrics_test.cc)
kaminpar_add_shm_test(test_shm_graph kaminpar/graph_test.cc)
kaminpar_add_shm_test(test_subgraph_extraction kaminpar/subgraph_extraction_test.cc)
kaminpar_add_shm_test(test_graph_utils kaminpar/graph_utils_test.cc)

if (KAMINPAR_BUILD_DISTRIBUTED)
    include(cmake/KaTestrophe.cmake)

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
            common 
            kaminpar 
            dkaminpar)
        target_include_directories(${KAMINPAR_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        katestrophe_add_mpi_test(${KAMINPAR_TARGET_NAME} CORES ${KAMINPAR_CORES})
    
        message(STATUS "Enabled distributed test: ${KAMINPAR_TARGET_NAME}, PEs: ${KAMINPAR_CORES}")
    endfunction()

    # dKaMinPar 
    kaminpar_add_dist_test(test_dist_global_move_conflict_resolver
        FILES dkaminpar/refinement/move_conflict_resolver_test.cc
        CORES 1 4)

    # dKaMinPar -> Graph Utils
    kaminpar_add_dist_test(test_dist_graphutils_block_extractor
        FILES dkaminpar/graphutils/block_extractor_test.cc
        CORES 1 4)

    kaminpar_add_dist_test(test_dist_graphutils_rearrangement
        FILES dkaminpar/graphutils/rearrangement_test.cc
        CORES 1 4)

    kaminpar_add_dist_test(test_dist_graphutils_bfs_extractor
        FILES dkaminpar/graphutils/bfs_extractor_test.cc
        CORES 1 4)

    kaminpar_add_dist_test(test_dist_graphutils_allgather
        FILES dkaminpar/graphutils/allgather_test.cc
        CORES 1 4)

    kaminpar_add_dist_test(test_dist_graphutils_cluster_contraction
        FILES dkaminpar/graphutils/cluster_contraction_test.cc
        CORES 1)

    # dKaMinPar -> Algorithms 
    kaminpar_add_dist_test(test_dist_greedy_node_coloring
        FILES dkaminpar/algorithms/greedy_node_coloring_test.cc
        CORES 1 4)

    kaminpar_add_dist_test(test_dist_independent_set
        FILES dkaminpar/algorithms/independent_set_test.cc
        CORES 1 4)

    # dKaMinPar -> MPI
    kaminpar_add_dist_test(test_dist_sparse_alltoall
        FILES dkaminpar/mpi/sparse_alltoall_test.cc
        CORES 1 4 11)

    kaminpar_add_dist_test(test_dist_graph_topology
        FILES dkaminpar/mpi/grid_topology_test.cc
        CORES 1)

    kaminpar_add_dist_test(test_dist_sparse_allreduce
        FILES dkaminpar/mpi/sparse_allreduce_test.cc
        CORES 1 4 8)
endif ()

