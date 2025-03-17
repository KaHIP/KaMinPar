set(COMMON_IO_SOURCE_FILES
    util/file_toker.h
    util/binary_util.h)

set(KAMINPAR_IO_SOURCE_FILES
    kaminpar_io.h kaminpar_io.cc
    metis_parser.h metis_parser.cc
    parhip_parser.h parhip_parser.cc
    graph_compression_binary.h graph_compression_binary.cc
    ${COMMON_IO_SOURCE_FILES})

set(DKAMINPAR_IO_SOURCE_FILES
    dist_io.h dist_io.cc
    dist_metis_parser.h dist_metis_parser.cc
    dist_parhip_parser.h dist_parhip_parser.cc
    dist_skagen.h dist_skagen.cc
    ${COMMON_IO_SOURCE_FILES})

add_library(kaminpar_io ${KAMINPAR_IO_SOURCE_FILES})
add_library(KaMinPar::KaMinParIO ALIAS kaminpar_io)

target_include_directories(kaminpar_io PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
)
target_link_libraries(kaminpar_io PUBLIC kaminpar_shm)
target_compile_options(kaminpar_io PRIVATE ${KAMINPAR_WARNING_FLAGS})

if (TARGET kaminpar_dist)
    add_library(kaminpar_dist_io ${DKAMINPAR_IO_SOURCE_FILES})
    add_library(KaMinPar::dKaMinParIO ALIAS kaminpar_dist_io)

    target_link_libraries(kaminpar_dist_io PUBLIC kaminpar_dist KaGen::KaGen)
    target_compile_options(kaminpar_dist_io PRIVATE ${KAMINPAR_WARNING_FLAGS})
endif ()
