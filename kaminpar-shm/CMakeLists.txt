file(GLOB_RECURSE KAMINPAR_SHM_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(KaMinPar ${KAMINPAR_SHM_SOURCE_FILES})
add_library(KaMinPar::KaMinPar ALIAS KaMinPar)

target_include_directories(KaMinPar PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/kaminpar-shm/>)

target_link_libraries(KaMinPar PUBLIC KaMinPar::KaMinParCommon)
target_compile_options(KaMinPar PRIVATE ${KAMINPAR_WARNING_FLAGS})

if (Sparsehash_FOUND)
    target_link_libraries(KaMinPar PUBLIC Sparsehash::Sparsehash)
endif ()

if (KAMINPAR_BUILD_WITH_WHFC)
    target_include_directories(KaMinPar PRIVATE $<BUILD_INTERFACE:${whfc_SOURCE_DIR}>)
endif ()

# If we can find Mt-KaHyPar, make it available as an option for refinement
if (KAMINPAR_BUILD_WITH_MTKAHYPAR)
    target_compile_definitions(KaMinPar PRIVATE -DKAMINPAR_HAVE_MTKAHYPAR_LIB)
    target_link_libraries(KaMinPar PUBLIC mtkahypar)
endif ()
