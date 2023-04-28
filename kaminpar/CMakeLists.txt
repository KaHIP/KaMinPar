file(GLOB_RECURSE SHM_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(kaminpar ${SHM_SOURCE_FILES})
target_include_directories(kaminpar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_link_libraries(kaminpar PUBLIC common)

# If we can find Mt-KaHyPar, make it available as an option for refinement
if (KAMINPAR_BUILD_WITH_MTKAHYPAR)
    target_compile_definitions(kaminpar PRIVATE -DKAMINPAR_HAVE_MTKAHYPAR_LIB)
    target_link_libraries(kaminpar PUBLIC mtkahypar)
endif ()
