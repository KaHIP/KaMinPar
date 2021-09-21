set(KAMINPAR_SOURCE_FILES
  algorithm/parallel_label_propagation.h
  algorithm/graph_contraction.cc
  algorithm/graph_contraction.h
  algorithm/graph_extraction.cc
  algorithm/graph_extraction.h
  algorithm/graph_permutation.cc
  algorithm/graph_permutation.h
  coarsening/i_coarsener.h
  coarsening/parallel_label_propagation_coarsener.h
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
  refinement/parallel_balancer.cc
  refinement/parallel_balancer.h
  refinement/parallel_label_propagation_refiner.h
  utility/strings.cc
  utility/strings.h
  utility/timer.cc
  utility/timer.h
  utility/random.cc
  utility/random.h
  utility/metrics.cc
  utility/metrics.h
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
  factories.cc
  factories.h
  parallel.h)

add_library(kaminpar_impl OBJECT ${KAMINPAR_SOURCE_FILES})
target_include_directories(kaminpar_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link TBB
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)
target_link_libraries(kaminpar_impl PUBLIC ${TBB_LIBRARIES})
target_include_directories(kaminpar_impl PUBLIC ${TBB_INCLUDE_DIRS})

# Link libnuma if available
find_library(NUMA_LIB numa)
if (NUMA_LIB)
    target_link_libraries(kaminpar_impl PUBLIC ${NUMA_LIB})
endif ()