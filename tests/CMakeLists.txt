file(COPY test_instances DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

function(add_kaminpar_test target)
    add_gmock_test(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE shm_partitioner_base)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if (KAMINPAR_BACKWARD_CPP)
        add_backward(${target})
    endif ()
endfunction()

# Datastructures
add_kaminpar_test(static_array_test datastructure/static_array_test.cc)
add_kaminpar_test(graph_utils_test datastructure/graph_utils_test.cc)
add_kaminpar_test(fast_reset_array_test datastructure/fast_reset_array_test.cc)
add_kaminpar_test(binary_heap_test datastructure/binary_heap_test.cc)
add_kaminpar_test(graph_test datastructure/graph_test.cc)
add_kaminpar_test(marker_test datastructure/marker_test.cc)
add_kaminpar_test(subgraph_extraction_test datastructure/subgraph_extraction_test.cc)

# Utility code
add_kaminpar_test(math_test utility/math_test.cc)
add_kaminpar_test(utility_test utility/utility_test.cc)
add_kaminpar_test(metrics_test utility/metrics_test.cc)

# Parallel utility code
add_kaminpar_test(test_parallel_algorithm parallel/algorithm_test.cc)

# IO
add_kaminpar_test(io_test io_test.cc)
