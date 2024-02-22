include(GoogleTest)

function(kaminpar_add_shm_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE 
        GTest::gtest_main 
        GTest::gmock_main
        kaminpar_common
        kaminpar_shm)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    gtest_discover_tests(${target})

    message(STATUS "Enabled shared-memory test: ${target}")
endfunction()

function(kaminpar_add_common_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE 
        GTest::gtest_main
        GTest::gmock_main
        kaminpar_common)
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
kaminpar_add_common_test(test_common_dynamic_map common/datastructures/dynamic_map_test.cc)
kaminpar_add_common_test(test_common_compact_hash_map common/datastructures/compact_hash_map_test.cc)

# Common -> Misc
kaminpar_add_common_test(test_common_math common/math_test.cc)
kaminpar_add_common_test(test_common_string common/strutils_test.cc)
kaminpar_add_common_test(test_common_parallel_algorithm common/parallel/algorithm_test.cc)

# KaMinPar -> End-to-end
kaminpar_add_shm_test(test_shm_endtoend endtoend/shm_endtoend_test.cc)

# KaMinPar -> Utils
kaminpar_add_shm_test(test_shm_metrics shm/metrics_test.cc)
kaminpar_add_shm_test(test_shm_subgraph_extraction shm/subgraph_extraction_test.cc)
kaminpar_add_shm_test(test_shm_graph_utils shm/graph_utils_test.cc)
kaminpar_add_shm_test(test_shm_partition_utils shm/partition_utils_test.cc)

# KaMinPar -> Data structures
kaminpar_add_shm_test(test_shm_graph shm/graph_test.cc)
kaminpar_add_shm_test(test_shm_delta_partitioned_graph shm/datastructures/delta_partitioned_graph_test.cc)

if (KAMINPAR_BUILD_DISTRIBUTED)
    include(cmake/KaTestrophe.cmake)

    function(kaminpar_add_mpi_test KAMINPAR_TARGET_NAME)
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
                kaminpar_common
                kaminpar_shm
                kaminpar_mpi)
        target_include_directories(${KAMINPAR_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        katestrophe_add_mpi_test(${KAMINPAR_TARGET_NAME} CORES ${KAMINPAR_CORES})

        message(STATUS "Enabled MPI test: ${KAMINPAR_TARGET_NAME}, PEs: ${KAMINPAR_CORES}")
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
            kaminpar_common
            kaminpar_shm
            kaminpar_mpi
            kaminpar_dist)
        target_include_directories(${KAMINPAR_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        katestrophe_add_mpi_test(${KAMINPAR_TARGET_NAME} CORES ${KAMINPAR_CORES})
    
        message(STATUS "Enabled distributed test: ${KAMINPAR_TARGET_NAME}, PEs: ${KAMINPAR_CORES}")
    endfunction()

    # dKaMinPar -> End-to-end
    kaminpar_add_dist_test(test_dist_endtoend 
        FILES endtoend/dist_endtoend_test.cc
        CORES 1 2 4)

    # dKaMinPar -> Coarsening
    kaminpar_add_dist_test(test_dist_coarsening_cluster_contraction
        FILES dist/coarsening/cluster_contraction_test.cc
        CORES 1)

    kaminpar_add_dist_test(test_dist_coarsening_cluster_contraction_internal
        FILES dist/coarsening/internal_cluster_contraction_test.cc
        CORES 1)

    # dKaMinPar -> Graph Utils
    kaminpar_add_dist_test(test_dist_graphutils_block_extractor
        FILES dist/graphutils/block_extractor_test.cc
        CORES 1 4)

    kaminpar_add_dist_test(test_dist_graphutils_rearrangement
        FILES dist/graphutils/rearrangement_test.cc
        CORES 1 4)

    kaminpar_add_dist_test(test_dist_graphutils_bfs_extractor
        FILES dist/graphutils/bfs_extractor_test.cc
        CORES 1 4)

    kaminpar_add_dist_test(test_dist_graphutils_allgather
        FILES dist/graphutils/allgather_test.cc
        CORES 1 4)

    # dKaMinPar -> Algorithms 
    kaminpar_add_dist_test(test_dist_greedy_node_coloring
        FILES dist/algorithms/greedy_node_coloring_test.cc
        CORES 1 4)

    kaminpar_add_dist_test(test_dist_independent_set
        FILES dist/algorithms/independent_set_test.cc
        CORES 1 4)

    #  MPI
    kaminpar_add_mpi_test(test_mpi_sparse_alltoall
        FILES mpi/sparse_alltoall_test.cc
        CORES 1 4 11)

    kaminpar_add_mpi_test(test_mpi_graph_topology
        FILES mpi/grid_topology_test.cc
        CORES 1)

    kaminpar_add_mpi_test(test_mpi_sparse_allreduce
        FILES mpi/sparse_allreduce_test.cc
        CORES 1 4 8)
endif ()

