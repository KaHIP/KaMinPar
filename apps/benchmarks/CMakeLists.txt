function(add_shm_benchmark target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE 
        KaMinPar::KaMinPar 
        KaMinPar::KaMinParCLI11 
        KaGen::KaGen)
    message(STATUS "Enabled benchmark: ${target}")
endfunction()

# Shared-memory benchmarks
add_shm_benchmark(shm_refinement_benchmark shm_refinement_benchmark.cc)
add_shm_benchmark(shm_gain_cache_benchmark shm_gain_cache_benchmark.cc)

if (KAMINPAR_BUILD_DISTRIBUTED)
    function(add_dist_benchmark target)
        add_shm_benchmark(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE 
            KaMinPar::dKaMinPar 
            KaMinPar::dKaMinParCLI11)
    endfunction()

    # Distributed benchmarks
    add_dist_benchmark(dist_coloring_benchmark dist_coloring_benchmark.cc)
    add_dist_benchmark(dist_contraction_benchmark dist_contraction_benchmark.cc)
    add_dist_benchmark(dist_refinement_benchmark dist_refinement_benchmark.cc)
    add_dist_benchmark(dist_coarsening_benchmark dist_coarsening_benchmark.cc)
    add_dist_benchmark(dist_block_clustering_benchmark dist_block_clustering_benchmark.cc)
endif ()
