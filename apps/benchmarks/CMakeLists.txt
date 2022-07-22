function(add_benchmark target)
    add_executable(${target} ${ARGN} "${CMAKE_CURRENT_BINARY_DIR}/../environment.cc")
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE common_base shm_partitioner_base)

    message(STATUS "Added benchmark ${target}.")
endfunction()

function(add_distributed_benchmark target)
    if (TARGET dist_partitioner_base)
        add_benchmark(${target} ${ARGN} ${DKAMINPAR_SOURCE_FILES})
        target_link_libraries(${target} PRIVATE dist_partitioner_base)

        if (KAMINPAR_ENABLE_GRAPHGEN)
            target_link_libraries(${target} PUBLIC kagen_library)
        endif ()
    else ()
        message(STATUS "Cannot add distributed app: distributed library is not available, presumably because MPI"
                " could not be found.")
    endif ()
endfunction()

add_benchmark(coarsening_benchmark coarsening_benchmark.cc)
add_benchmark(balancing_benchmark balancing_benchmark.cc)

add_distributed_benchmark(dcoarsening_benchmark dcoarsening_benchmark.cc)
add_distributed_benchmark(dbalancing_benchmark dbalancing_benchmark.cc)
add_distributed_benchmark(dcontraction_benchmark dcontraction_benchmark.cc)

if (KAMINPAR_ENABLE_GRAPHGEN)
    add_distributed_benchmark(graphgen_benchmark graphgen_benchmark.cc)
endif ()
