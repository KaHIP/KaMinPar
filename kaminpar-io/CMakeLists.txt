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

add_library(KaMinParIO ${KAMINPAR_IO_SOURCE_FILES})
add_library(KaMinPar::KaMinParIO ALIAS KaMinParIO)

target_include_directories(KaMinParIO PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/kaminpar-io/>)
target_link_libraries(KaMinParIO PUBLIC KaMinPar::KaMinPar)
target_compile_options(KaMinParIO PRIVATE ${KAMINPAR_WARNING_FLAGS})

if (TARGET dKaMinPar)
    add_library(dKaMinParIO ${DKAMINPAR_IO_SOURCE_FILES})
    add_library(KaMinPar::dKaMinParIO ALIAS dKaMinParIO)

    target_link_libraries(dKaMinParIO PUBLIC KaMinPar::dKaMinPar KaGen::KaGen)
    target_compile_options(dKaMinParIO PRIVATE ${KAMINPAR_WARNING_FLAGS})
endif ()
