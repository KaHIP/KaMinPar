# Make commit hash available
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)

# Make hostname available
include(GetHostname)
GetHostname(HOSTNAME)

# Make list of modified files available
include(GetModifiedGitFiles)
GetModifiedGitFiles(MODIFIED_FILES)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/environment.cc.in" "${CMAKE_CURRENT_BINARY_DIR}/environment.cc" @ONLY)

file(GLOB_RECURSE KAMINPAR_COMMON_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(KaMinParCommon OBJECT ${KAMINPAR_COMMON_SOURCE_FILES} "${CMAKE_CURRENT_BINARY_DIR}/environment.cc")
add_library(KaMinPar::KaMinParCommon ALIAS KaMinParCommon)

target_compile_definitions(KaMinParCommon PUBLIC ${KAMINPAR_DEFINITIONS})
target_compile_options(KaMinParCommon PRIVATE ${KAMINPAR_WARNING_FLAGS})

target_link_libraries(KaMinParCommon PUBLIC TBB::tbb)
if (KAMINPAR_BUILD_WITH_KASSERT)
    target_link_libraries(KaMinParCommon PUBLIC kassert::kassert)
    set_target_properties(KaMinParCommon PROPERTIES KASSERT_ASSERTION_LEVEL ${KAMINPAR_ASSERTION_LEVEL_CODE})
endif ()
if (KAMINPAR_ENABLE_TBB_MALLOC)
    target_link_libraries(KaMinParCommon PUBLIC TBB::tbbmalloc)
endif ()

if (NUMA_LIB)
    target_link_libraries(KaMinParCommon PUBLIC ${NUMA_LIB})
endif ()
