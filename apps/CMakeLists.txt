set(KAMINPAR_IO_SOURCE_FILES
    io/file_tokener.h
    io/binary_util.h
    io/shm_metis_parser.h
    io/shm_metis_parser.cc
    io/shm_parhip_parser.h
    io/shm_parhip_parser.cc
    io/shm_compressed_graph_binary.h
    io/shm_compressed_graph_binary.cc
    io/shm_input_validator.h
    io/shm_input_validator.cc
    io/shm_io.h
    io/shm_io.cc)

add_library(kaminpar_io ${KAMINPAR_IO_SOURCE_FILES})
target_include_directories(kaminpar_io PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/io/../")
target_link_libraries(kaminpar_io PUBLIC KaMinPar::KaMinPar KaMinPar::KaMinParCLI)

add_library(KaMinPar::KaMinParIO ALIAS kaminpar_io)

function(add_shm_app target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE KaMinPar::KaMinPar KaMinPar::KaMinParCLI KaMinPar::KaMinParIO)
    install(TARGETS ${target})
    message(STATUS "Enabled app: ${target}")
endfunction()

function(add_dist_app target)
    if (TARGET kaminpar_dist)
        add_shm_app(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE KaMinPar::dKaMinPar KaMinPar::dKaMinParCLI)
    else ()
        message(STATUS "Disabled app: ${target}")
    endif ()
endfunction()

add_shm_app(KaMinPar KaMinPar.cc)

if (TARGET kaminpar_dist)
    add_dist_app(dKaMinPar dKaMinPar.cc)
    target_sources(dKaMinPar PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_io.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_metis_parser.h
        ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_metis_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_parhip_parser.h
        ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_parhip_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_skagen.h
        ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_skagen.cc)
    target_link_libraries(dKaMinPar PRIVATE KaGen::KaGen)
endif ()

if (KAMINPAR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()

if (KAMINPAR_BUILD_TOOLS)
    add_subdirectory(tools)
endif ()
