set(DKAMINPAR_SOURCE_FILES
        datastructure/distributed_graph.cc
        datastructure/distributed_graph.h
        distributed_io.cc
        distributed_io.h
        utility/distributed_math.h
        distributed_definitions.h)

# Only create target if MPI is available
find_package(MPI)

if (MPI_FOUND)
    add_library(dkaminpar_impl OBJECT ${DKAMINPAR_SOURCE_FILES})
    target_include_directories(dkaminpar_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(dkaminpar_impl PUBLIC kaminpar_impl MPI::MPI_CXX)
else ()
    message(STATUS "MPI is NOT available; distributed target will not be available!")
endif ()
