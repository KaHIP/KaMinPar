function(add_benchmark target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE kaminpar_impl)
    message(STATUS "Benchmark ${target}")
endfunction()

add_benchmark(coarsening_benchmark coarsening_benchmark.cc)
