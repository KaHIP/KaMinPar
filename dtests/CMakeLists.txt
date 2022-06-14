list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(KaTestrophe)
include(dKaMinParTestHelper)
include(GoogleTest)

file(COPY test_instances DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Coarsening
dkaminpar_register_mpi_test(dtest_global_label_propagation_clustering
        FILES coarsening/test_global_label_propagation_clustering.cc
        CORES 3)
dkaminpar_register_mpi_test(dtest_local_contraction
        FILES coarsening/test_local_contraction.cc
        CORES 3)
dkaminpar_register_mpi_test(dtest_global_clustering_contraction
        FILES coarsening/test_global_clustering_contraction.cc
        CORES 3)
dkaminpar_register_mpi_test(dtest_global_clustering_projection
        FILES coarsening/test_global_clustering_projection.cc
        CORES 3)

# Graph utils
dkaminpar_register_mpi_test(dtest_rearrange_graph
        FILES graphutils/rearrange_graph_test.cc
        CORES 3)

dkaminpar_register_mpi_test(dtest_extract_subgraphs
        FILES graphutils/extract_subgraphs_test.cc
        CORES 3)

# MPI wrapper
dkaminpar_register_mpi_test(dtest_mpi_graph
        FILES test_mpi_graph.cc
        CORES 3)
dkaminpar_register_mpi_test(dtest_sparse_alltoall 
        FILES sparse_alltoall_test.cc 
        CORES 1 2 3 4 5 6 7 8)

# Utility functions
dkaminpar_register_test(dtest_math
        FILES utility/test_math.cc)

