set(DKAMINPAR_SOURCE_FILES
        graphutils/allgather_graph.cc
        graphutils/allgather_graph.h
        graphutils/rearrange_graph.cc
        graphutils/rearrange_graph.h
        coarsening/contraction_helper.h
        coarsening/local_clustering_contraction.cc
        coarsening/local_clustering_contraction.h
        coarsening/global_clustering_contraction.cc
        coarsening/global_clustering_contraction.h
        coarsening/i_clustering.h
        coarsening/local_label_propagation_clustering.cc
        coarsening/local_label_propagation_clustering.h
        coarsening/locking_label_propagation_clustering.cc
        coarsening/locking_label_propagation_clustering.h
        coarsening/global_label_propagation_clustering.cc
        coarsening/global_label_propagation_clustering.h
        datastructure/distributed_graph.cc
        datastructure/distributed_graph.h
        datastructure/distributed_graph_builder.h
        initial_partitioning/i_initial_partitioner.h
        initial_partitioning/kaminpar_initial_partitioner.cc
        initial_partitioning/kaminpar_initial_partitioner.h
        initial_partitioning/random_initial_partitioner.cc
        initial_partitioning/random_initial_partitioner.h
        distributed_io.cc
        distributed_io.h
        utility/math.h
        utility/distributed_timer.cc
        utility/distributed_timer.h
        mpi_wrapper.h
        mpi_graph.h
        utility/metrics.cc
        utility/metrics.h
        utility/vector_ets.h
        refinement/i_distributed_refiner.h
        refinement/noop_refiner.cc
        refinement/noop_refiner.h
        refinement/distributed_probabilistic_label_propagation_refiner.cc
        refinement/distributed_probabilistic_label_propagation_refiner.h
        partitioning_scheme/kway.cc
        partitioning_scheme/kway.h
        partitioning_scheme/partitioning.cc
        partitioning_scheme/partitioning.h
        definitions.h
        context.cc
        context.h
        factories.cc
        factories.h
        growt.h)

find_package(OpenMP REQUIRED)

add_library(dkaminpar_impl OBJECT ${DKAMINPAR_SOURCE_FILES})
target_include_directories(dkaminpar_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dkaminpar_impl PUBLIC kaminpar_impl MPI::MPI_CXX OpenMP::OpenMP_CXX growt)

if (KAMINPAR_BACKWARD_CPP)
    add_backward(dkaminpar_impl)
endif ()
