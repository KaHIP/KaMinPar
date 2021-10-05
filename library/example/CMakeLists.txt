add_executable(library_example_from_file library_example_from_file.cc)
target_link_libraries(library_example_from_file kaminpar)

add_executable(library_example_static library_example_static.cc)
target_link_libraries(library_example_static kaminpar)

if (KAMINPAR_USE_BACKWARD_CPP)
    add_backward(library_example_from_file)
    add_backward(library_example_static)
endif ()