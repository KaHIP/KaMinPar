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

add_library(kaminpar_common OBJECT ${KAMINPAR_COMMON_SOURCE_FILES} "${CMAKE_CURRENT_BINARY_DIR}/environment.cc")
target_include_directories(kaminpar_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_compile_definitions(kaminpar_common PUBLIC ${KAMINPAR_DEFINITIONS})

# Link libraries
find_package(TBB REQUIRED) 
find_library(NUMA_LIB numa) # optional 

target_link_libraries(kaminpar_common PUBLIC TBB::tbb TBB::tbbmalloc kassert::kassert)
if (NUMA_LIB) 
    target_link_libraries(kaminpar_common PUBLIC ${NUMA_LIB})
endif ()

