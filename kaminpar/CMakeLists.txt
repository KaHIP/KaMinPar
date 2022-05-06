file(GLOB_RECURSE SHM_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(shm_partitioner_base OBJECT ${SHM_SOURCE_FILES})
target_include_directories(shm_partitioner_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link TBB
find_package(TBB REQUIRED)
target_link_libraries(shm_partitioner_base PUBLIC TBB::tbb TBB::tbbmalloc)

# Link kassert 
target_link_libraries(shm_partitioner_base PUBLIC kassert::kassert)

# Link libnuma if available
find_library(NUMA_LIB numa)
if (NUMA_LIB)
    target_link_libraries(shm_partitioner_base PUBLIC ${NUMA_LIB})
endif ()
