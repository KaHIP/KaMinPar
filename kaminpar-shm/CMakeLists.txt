file(GLOB_RECURSE KAMINPAR_SHM_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(KaMinPar ${KAMINPAR_SHM_SOURCE_FILES})
add_library(KaMinPar::KaMinPar ALIAS KaMinPar)

target_include_directories(KaMinPar PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/kaminpar-shm/>)
target_link_libraries(KaMinPar PUBLIC KaMinPar::KaMinParCommon)
target_compile_options(KaMinPar PRIVATE ${KAMINPAR_WARNING_FLAGS})
set_target_properties(KaMinPar PROPERTIES KASSERT_ASSERTION_LEVEL ${KAMINPAR_ASSERTION_LEVEL_CODE})

if (KAMINPAR_BUILD_WITH_SPARSEHASH)
    target_link_libraries(KaMinPar PRIVATE Sparsehash::Sparsehash)
endif ()

if (KAMINPAR_BUILD_WITH_MTKAHYPAR)
    target_link_libraries(KaMinPar PRIVATE MtKaHyPar::mtkahypar)
endif ()
