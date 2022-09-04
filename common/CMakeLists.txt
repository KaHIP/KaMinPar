file(GLOB_RECURSE COMMON_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(common_base OBJECT ${COMMON_SOURCE_FILES})
target_include_directories(common_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
find_package(TBB REQUIRED) 
find_library(NUMA_LIB numa) # optional 

target_link_libraries(common_base PUBLIC TBB::tbb TBB::tbbmalloc kassert::kassert)
if (NUMA_LIB) 
    target_link_libraries(common_base PUBLIC ${NUMA_LIB})
endif ()

