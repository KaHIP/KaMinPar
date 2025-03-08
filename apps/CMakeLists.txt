set(COMMON_APP_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/version.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/version.h)

function(add_shm_app target)
    add_executable(${target} ${ARGN} ${COMMON_APP_SOURCE_FILES})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE KaMinPar::KaMinPar KaMinPar::KaMinParCLI KaMinPar::KaMinParIO)
    target_compile_options(${target} PRIVATE ${KAMINPAR_WARNING_FLAGS})
    install(TARGETS ${target})
    message(STATUS "Enabled app: ${target}")
endfunction()

function(add_dist_app target)
    if (TARGET kaminpar_dist)
        add_shm_app(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE KaMinPar::dKaMinPar KaMinPar::dKaMinParCLI KaMinPar::dKaMinParIO KaGen::KaGen)
    else ()
        message(STATUS "Disabled app: ${target}")
    endif ()
endfunction()

add_shm_app(KaMinPar KaMinPar.cc)

if (TARGET kaminpar_dist)
    add_dist_app(dKaMinPar dKaMinPar.cc)

    if (KAMINPAR_BUILD_WITH_BACKWARD)
        target_link_libraries(dKaMinPar PRIVATE Backward::Backward)
        target_compile_definitions(dKaMinPar PRIVATE -DKAMINPAR_HAVE_BACKWARD)
    endif ()
endif ()

if (KAMINPAR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()

if (KAMINPAR_BUILD_TOOLS)
    add_subdirectory(tools)
endif ()
