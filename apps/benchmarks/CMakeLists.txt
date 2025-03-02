function(add_shm_benchmark target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE
        KaMinPar::KaMinPar
        KaMinPar::KaMinParCLI
        KaMinPar::KaMinParIO
        KaGen::KaGen)
    target_compile_options(${target} PRIVATE ${KAMINPAR_WARNING_FLAGS})
    message(STATUS "Enabled benchmark: ${target}")
endfunction()

# Shared-memory benchmarks
add_shm_benchmark(shm_input_benchmark shm_input_benchmark.cc)
add_shm_benchmark(shm_label_propagation_benchmark shm_label_propagation_benchmark.cc)
add_shm_benchmark(shm_compressed_graph_benchmark shm_compressed_graph_benchmark.cc)

if (KAMINPAR_BUILD_DISTRIBUTED)
    function(add_dist_benchmark target)
        add_shm_benchmark(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE
            KaMinPar::dKaMinPar
            KaMinPar::dKaMinParCLI
            KaMinPar::dKaMinParIO)
    endfunction()

    # Distributed benchmarks
    # <...>
endif ()
