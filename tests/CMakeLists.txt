file(COPY test_instances DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

function(add_kaminpar_test target)
    add_gmock_test(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE shm_partitioner_base)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if (KAMINPAR_BACKWARD_CPP)
        add_backward(${target})
    endif ()
endfunction()

function(add_common_test target)
    add_gmock_test(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE common_base)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

# Datastructures
add_kaminpar_test(test_static_array datastructure/static_array_test.cc)
add_kaminpar_test(test_graph_utils datastructure/graph_utils_test.cc)
add_kaminpar_test(test_fast_reset_array datastructure/fast_reset_array_test.cc)
add_kaminpar_test(test_binary_heap datastructure/binary_heap_test.cc)
add_kaminpar_test(test_graph datastructure/graph_test.cc)
add_kaminpar_test(test_marker datastructure/marker_test.cc)
add_kaminpar_test(test_subgraph_extraction datastructure/subgraph_extraction_test.cc)

# Utility code
add_kaminpar_test(test_utils_math utils/math_test.cc)
add_kaminpar_test(test_utils_string utils/string_test.cc)
add_kaminpar_test(test_utils_metrics utils/metrics_test.cc)

# Parallel code
add_kaminpar_test(test_parallel_algorithm parallel/algorithm_test.cc)

# IO code
add_kaminpar_test(test_io io_test.cc)

# Common 
add_common_test(test_common_utils_preallocated_vector common/utils/preallocated_vector_test.cc)
