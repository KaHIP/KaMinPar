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

file(GLOB_RECURSE COMMON_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(common_base OBJECT ${COMMON_SOURCE_FILES} "${CMAKE_CURRENT_BINARY_DIR}/environment.cc")
target_include_directories(common_base PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_compile_definitions(common_base PUBLIC ${KAMINPAR_DEFINITIONS})

# Link libraries
find_package(TBB REQUIRED) 
find_library(NUMA_LIB numa) # optional 

target_link_libraries(common_base PUBLIC TBB::tbb TBB::tbbmalloc kassert::kassert)
if (NUMA_LIB) 
    target_link_libraries(common_base PUBLIC ${NUMA_LIB})
endif ()

