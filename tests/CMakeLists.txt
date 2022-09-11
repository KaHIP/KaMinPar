list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(TestHelper)

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
kaminpar_add_common_test(test_common_string common/string_test.cc)
kaminpar_add_common_test(test_common_parallel_algorithm common/parallel/algorithm_test.cc)

# dKaMinPar -> Graph Utils
kaminpar_add_dist_test(test_dist_graph_extraction
    FILES dkaminpar/graphutils/graph_extraction_test.cc
    CORES 1 4 16)

kaminpar_add_dist_test(test_dist_graph_rearrangement
    FILES dkaminpar/graphutils/graph_rearrangement_test.cc
    CORES 1 4 16)

kaminpar_add_dist_test(test_dist_independent_set
    FILES dkaminpar/graphutils/independent_set_test.cc
    CORES 1 4 16)

kaminpar_add_dist_test(test_dist_bfs_extractor
    FILES dkaminpar/graphutils/bfs_extractor_test.cc
    CORES 1 2 4 8 16)

kaminpar_add_dist_test(test_dist_global_move_conflict_resolver
    FILES dkaminpar/refinement/move_conflict_resolver_test.cc
    CORES 1 2 3 4 5 6 7 8)

kaminpar_add_dist_test(test_dist_sparse_alltoall
    HEADER_ONLY
    FILES dkaminpar/sparse_alltoall_test.cc
    CORES 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 20 21 22 23 24 32 64)

kaminpar_add_dist_test(test_dist_graph_topology
    HEADER_ONLY
    FILES dkaminpar/mpi/grid_topology_test.cc
    CORES 1)

# KaMinPar 
kaminpar_add_shm_test(test_shm_io kaminpar/io_test.cc)

