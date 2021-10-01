set(DKAMINPAR_SOURCE_FILES
        application/arguments.cc
        application/arguments.h
        algorithm/distributed_graph_contraction.cc
        algorithm/distributed_graph_contraction.h
        algorithm/allgather_graph.cc
        algorithm/allgather_graph.h
        algorithm/seq_global_graph_contraction_redistribution.cc
        algorithm/seq_global_graph_contraction_redistribution.h
        coarsening/i_clustering.h
        coarsening/distributed_local_label_propagation_clustering.h
        coarsening/distributed_local_label_propagation_clustering.cc
        coarsening/distributed_global_label_propagation_clustering.cc
        coarsening/distributed_global_label_propagation_clustering.h
        coarsening/locking_lp_clustering.cc
        coarsening/locking_lp_clustering.h
        datastructure/distributed_graph.cc
        datastructure/distributed_graph.h
        datastructure/distributed_graph_builder.h
        initial_partitioning/i_initial_partitioner.h
        distributed_io.cc
        distributed_io.h
        utility/distributed_math.h
        utility/distributed_timer.cc
        utility/distributed_timer.h
        mpi_utils.h
        mpi_graph_utils.h
        utility/distributed_metrics.cc
        utility/distributed_metrics.h
        utility/vector_ets.h
        refinement/i_distributed_refiner.h
        refinement/distributed_probabilistic_label_propagation_refiner.cc
        refinement/distributed_probabilistic_label_propagation_refiner.h
        partitioning_scheme/kway.cc
        partitioning_scheme/kway.h
        partitioning_scheme/partitioning.cc
        partitioning_scheme/partitioning.h
        distributed_definitions.h
        distributed_context.cc
        distributed_context.h
        mpi_wrapper.h
        growt.h)

# Only create target if MPI is available
find_package(MPI)

if (MPI_FOUND)
    add_library(dkaminpar_impl OBJECT ${DKAMINPAR_SOURCE_FILES})
    target_include_directories(dkaminpar_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(dkaminpar_impl PUBLIC kaminpar_impl MPI::MPI_CXX growt)
else ()
    message(STATUS "MPI is NOT available; distributed target will not be available!")
endif ()
