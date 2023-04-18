function(add_benchmark target)
    add_executable(${target} ${ARGN} "${CMAKE_CURRENT_BINARY_DIR}/../environment.cc")
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE common kaminpar)

    message(STATUS "Enabled benchmark: ${target}")
endfunction()

function(add_distributed_benchmark target)
    if (TARGET dkaminpar)
        add_benchmark(${target} ${ARGN} "${CMAKE_CURRENT_SOURCE_DIR}/dist_benchmarks_common.cc")
        target_link_libraries(${target} PRIVATE dkaminpar)
    else ()
        message(STATUS "Disabled benchmark: ${target}")
    endif ()
endfunction()

add_benchmark(balancing_benchmark balancing_benchmark.cc)

add_distributed_benchmark(dist_label_propagation_benchmark dist_label_propagation_benchmark.cc)
add_distributed_benchmark(dist_coloring_benchmark dist_coloring_benchmark.cc)
add_distributed_benchmark(dist_flat_refinement_benchmark dist_flat_refinement_benchmark.cc)
add_distributed_benchmark(dist_contraction_benchmark dist_contraction_benchmark.cc)
add_distributed_benchmark(dist_balancing_benchmark dist_balancing_benchmark.cc)
add_distributed_benchmark(dist_coarsening_benchmark dist_coarsening_benchmark.cc)

