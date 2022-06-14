file(GLOB_RECURSE SHM_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(shm_partitioner_base OBJECT ${SHM_SOURCE_FILES})
target_include_directories(shm_partitioner_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(shm_partitioner_base PUBLIC common_base)
