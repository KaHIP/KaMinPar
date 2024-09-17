set(COMMON_IO_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/io/file_toker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/io/binary_util.h)

set(KAMINPAR_IO_SOURCE_FILES
    ${COMMON_IO_SOURCE_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_metis_parser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_metis_parser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_parhip_parser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_parhip_parser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_compressed_graph_binary.h
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_compressed_graph_binary.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_input_validator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_input_validator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_io.h
    ${CMAKE_CURRENT_SOURCE_DIR}/io/shm_io.cc)

set(DKAMINPAR_IO_SOURCE_FILES  
    ${COMMON_IO_SOURCE_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_io.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_metis_parser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_metis_parser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_parhip_parser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_parhip_parser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_skagen.h
    ${CMAKE_CURRENT_SOURCE_DIR}/io/dist_skagen.cc)

add_library(kaminpar_io ${KAMINPAR_IO_SOURCE_FILES})
target_include_directories(kaminpar_io PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(kaminpar_io PUBLIC KaMinPar::KaMinPar KaMinPar::KaMinParCLI)
target_compile_options(kaminpar_io PRIVATE ${KAMINPAR_WARNING_FLAGS})

add_library(KaMinPar::KaMinParIO ALIAS kaminpar_io)

function(add_shm_app target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE KaMinPar::KaMinPar KaMinPar::KaMinParCLI KaMinPar::KaMinParIO)
    target_compile_options(${target} PRIVATE ${KAMINPAR_WARNING_FLAGS})
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
    add_library(dkaminpar_io ${DKAMINPAR_IO_SOURCE_FILES})
    target_include_directories(dkaminpar_io PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(dkaminpar_io PUBLIC 
        KaMinPar::dKaMinPar 
        KaMinPar::dKaMinParCLI 
        KaGen::KaGen)
    target_compile_options(dkaminpar_io PRIVATE ${KAMINPAR_WARNING_FLAGS})

    add_library(KaMinPar::dKaMinParIO ALIAS dkaminpar_io)

    add_dist_app(dKaMinPar dKaMinPar.cc)
    target_link_libraries(dKaMinPar PRIVATE KaMinPar::dKaMinParIO)
endif ()

if (KAMINPAR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()

if (KAMINPAR_BUILD_TOOLS)
    add_subdirectory(tools)
endif ()
