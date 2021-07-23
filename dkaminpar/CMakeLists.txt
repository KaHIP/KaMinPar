set(DKAMINPAR_SOURCE_FILES
        application/arguments.h
        algorithm/distributed_graph_contraction.cc
        algorithm/distributed_graph_contraction.h
        algorithm/allgather_graph.cc
        algorithm/allgather_graph.h
        coarsening/distributed_local_label_propagation_coarsener.h
        datastructure/distributed_graph.cc
        datastructure/distributed_graph.h
        datastructure/distributed_graph_builder.h
        distributed_io.cc
        distributed_io.h
        utility/distributed_math.h
        mpi_utils.h
        mpi_graph_utils.h
        utility/distributed_metrics.cc
        utility/distributed_metrics.h
        utility/vector_ets.h
        refinement/distributed_label_propagation_refiner.h
        partitioning_scheme/kway.cc
        partitioning_scheme/kway.h
        partitioning_scheme/partitioning.cc
        partitioning_scheme/partitioning.h
        distributed_definitions.h
        distributed_context.cc
        distributed_context.h
        mpi_wrapper.h)

# Only create target if MPI is available
find_package(MPI)

if (MPI_FOUND)
    add_library(dkaminpar_impl OBJECT ${DKAMINPAR_SOURCE_FILES})
    target_include_directories(dkaminpar_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(dkaminpar_impl PUBLIC kaminpar_impl MPI::MPI_CXX)
else ()
    message(STATUS "MPI is NOT available; distributed target will not be available!")
endif ()
