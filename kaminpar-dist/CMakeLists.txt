file(GLOB_RECURSE KAMINPAR_DIST_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(dKaMinPar ${KAMINPAR_DIST_SOURCE_FILES})
add_library(KaMinPar::dKaMinPar ALIAS dKaMinPar)

target_include_directories(dKaMinPar PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/kaminpar-dist/>)
target_link_libraries(dKaMinPar PUBLIC KaMinPar::KaMinPar KaMinPar::KaMinParMPI growt)
target_compile_options(dKaMinPar PRIVATE ${KAMINPAR_WARNING_FLAGS})
set_target_properties(dKaMinPar PROPERTIES KASSERT_ASSERTION_LEVEL ${KAMINPAR_ASSERTION_LEVEL_CODE})

if (KAMINPAR_BUILD_WITH_SPARSEHASH)
    target_link_libraries(dKaMinPar PRIVATE Sparsehash::Sparsehash)
endif ()

if (KAMINPAR_BUILD_WITH_MTKAHYPAR)
    target_link_libraries(dKaMinPar PRIVATE MtKaHyPar::mtkahypar)
endif ()
