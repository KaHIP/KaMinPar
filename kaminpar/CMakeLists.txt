file(GLOB_RECURSE SHM_SOURCE_FILES CONFIGURE_DEPENDS
	*.cc *.h)

add_library(kaminpar ${SHM_SOURCE_FILES})
target_include_directories(kaminpar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_link_libraries(kaminpar PUBLIC common)
