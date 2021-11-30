set(KAMINPAR_SOURCE_FILES
        label_propagation.h
        graphutils/graph_contraction.cc
        graphutils/graph_contraction.h
        graphutils/graph_extraction.cc
        graphutils/graph_extraction.h
        graphutils/graph_permutation.h
        graphutils/graph_rearrangement.cc
        graphutils/graph_rearrangement.h
        application/arguments.cc
        application/arguments.h
        application/arguments_parser.h
        coarsening/i_clustering.h
        coarsening/i_coarsener.h
        coarsening/noop_coarsener.h
        coarsening/cluster_coarsener.h
        coarsening/cluster_coarsener.cc
        coarsening/label_propagation_clustering.h
        coarsening/label_propagation_clustering.cc
        datastructure/graph.cc
        datastructure/graph.h
        datastructure/binary_heap.h
        datastructure/fast_reset_array.h
        datastructure/rating_map.h
        datastructure/sparse_map.h
        datastructure/fixed_size_sparse_map.h
        datastructure/marker.h
        datastructure/queue.h
        datastructure/static_array.h
        datastructure/ts_navigable_linked_list.h
        initial_partitioning/bfs_bipartitioner.h
        initial_partitioning/bfs_bipartitioner.cc
        initial_partitioning/greedy_graph_growing_bipartitioner.cc
        initial_partitioning/greedy_graph_growing_bipartitioner.h
        initial_partitioning/i_bipartitioner.h
        initial_partitioning/pool_bipartitioner.h
        initial_partitioning/random_bipartitioner.h
        initial_partitioning/initial_coarsener.cc
        initial_partitioning/initial_coarsener.h
        initial_partitioning/sequential_graph_hierarchy.cc
        initial_partitioning/sequential_graph_hierarchy.h
        initial_partitioning/initial_refiner.cc
        initial_partitioning/initial_refiner.h
        initial_partitioning/initial_partitioning_facade.h
        initial_partitioning/seed_node_utils.cc
        initial_partitioning/seed_node_utils.h
        partitioning_scheme/parallel_recursive_bisection.cc
        partitioning_scheme/parallel_recursive_bisection.h
        partitioning_scheme/helper.cc
        partitioning_scheme/helper.h
        partitioning_scheme/parallel_initial_partitioner.cc
        partitioning_scheme/parallel_initial_partitioner.h
        partitioning_scheme/parallel_synchronized_initial_partitioner.cc
        partitioning_scheme/parallel_synchronized_initial_partitioner.h
        partitioning_scheme/parallel_simple_recursive_bisection.h
        partitioning_scheme/partitioning.cc
        partitioning_scheme/partitioning.h
        refinement/i_balancer.h
        refinement/i_refiner.h
        refinement/greedy_balancer.cc
        refinement/greedy_balancer.h
        refinement/label_propagation_refiner.h
        refinement/label_propagation_refiner.cc
        utility/strings.cc
        utility/strings.h
        utility/timer.cc
        utility/timer.h
        utility/random.cc
        utility/random.h
        utility/logger.cc
        utility/logger.h
        utility/console_io.cc
        utility/console_io.h
        utility/math.h
        utility/enum_string_conversion.h
        context.cc
        context.h
        io.cc
        io.h
        metrics.cc
        metrics.h
        factories.cc
        factories.h
        parallel.h
        definitions.cc
        definitions.h)

add_library(kaminpar_impl OBJECT ${KAMINPAR_SOURCE_FILES})
target_include_directories(kaminpar_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (KAMINPAR_BACKWARD_CPP)
    add_backward(kaminpar_impl)
endif ()

# Link TBB
find_package(TBB REQUIRED)
target_link_libraries(kaminpar_impl PUBLIC TBB::tbb TBB::tbbmalloc)

# Link libnuma if available
find_library(NUMA_LIB numa)
if (NUMA_LIB)
    target_link_libraries(kaminpar_impl PUBLIC ${NUMA_LIB})
endif ()
