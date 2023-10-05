function(add_shm_app target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE KaMinPar::KaMinPar KaMinPar::KaMinParCLI11)
    install(TARGETS ${target})
    message(STATUS "Enabled app: ${target}")
endfunction()

function(add_dist_app target)
    if (TARGET kaminpar_dist)
        add_shm_app(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE KaMinPar::dKaMinPar KaMinPar::dKaMinParCLI11)
    else ()
        message(STATUS "Disabled app: ${target}")
    endif ()
endfunction()

add_shm_app(KaMinPar KaMinPar.cc)
target_sources(KaMinPar PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_io.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_input_validator.cc)

if (TARGET kaminpar_dist)
    add_dist_app(dKaMinPar dKaMinPar.cc)
    target_sources(dKaMinPar PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_io.cc)
    target_link_libraries(dKaMinPar PRIVATE KaGen::KaGen)
endif ()

if (KAMINPAR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()

