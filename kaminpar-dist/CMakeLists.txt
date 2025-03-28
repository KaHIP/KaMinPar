file(GLOB_RECURSE KAMINPAR_DIST_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(dKaMinPar ${KAMINPAR_DIST_SOURCE_FILES})
add_library(KaMinPar::dKaMinPar ALIAS dKaMinPar)

target_include_directories(dKaMinPar PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/kaminpar-dist/>)
target_link_libraries(dKaMinPar PUBLIC KaMinPar::KaMinPar KaMinPar::KaMinParMPI growt)
target_compile_options(dKaMinPar PRIVATE ${KAMINPAR_WARNING_FLAGS})

if (Sparsehash_FOUND)
    target_link_libraries(dKaMinPar PUBLIC Sparsehash::Sparsehash)
endif ()

# If we can find Mt-KaHyPar, make it available as an option for initial partitioning
if (KAMINPAR_BUILD_WITH_MTKAHYPAR)
    target_compile_definitions(dKaMinPar PRIVATE -DKAMINPAR_HAVE_MTKAHYPAR_LIB)
    target_link_libraries(dKaMinPar PUBLIC mtkahypar)
endif ()
