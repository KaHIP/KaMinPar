function(add_shm_app target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE KaMinPar::KaMinPar)
    install(TARGETS ${target})
    message(STATUS "Enabled app: ${target}")
endfunction()

function(add_dist_app target)
    if (TARGET dist_partitioner_base)
        add_shm_app(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE KaMinPar::dKaMinPar)
    else ()
        message(STATUS "Disabled app: ${target}")
    endif ()
endfunction()

add_shm_app(KaMinPar KaMinPar.cc)
target_sources(KaMinPar PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_io.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_input_validator.cc)

if (TARGET dist_partitioner_base)
    set(KAGEN_NODEPS ON)
    set(KAGEN_BUILD_APPS OFF)
    set(KAGEN_BUILD_EXAMPLES OFF)
    FetchContent_MakeAvailable(KaGen)

    add_dist_app(dKaMinPar dKaMinPar.cc)
    target_link_libraries(dKaMinPar PRIVATE KaGen::KaGen)
endif ()

