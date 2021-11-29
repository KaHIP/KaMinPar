function(add_benchmark target)
    add_executable(${target} ${ARGN} "${CMAKE_CURRENT_BINARY_DIR}/../environment.cc")
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE kaminpar_impl)
    message(STATUS "Benchmark ${target}")

    if (KAMINPAR_BACKWARD_CPP)
        add_backward(${target})
    endif ()
endfunction()

function(add_distributed_benchmark target)
    if (TARGET dkaminpar_impl)
        add_benchmark(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE dkaminpar_impl)
    else ()
        message(STATUS "Cannot add distributed app: distributed library is not available, presumably because MPI"
                " could not be found.")
    endif ()
endfunction()

add_benchmark(coarsening_benchmark coarsening_benchmark.cc)
add_distributed_benchmark(dcoarsening_benchmark dcoarsening_benchmark.cc ../dkaminpar_arguments.cc)
