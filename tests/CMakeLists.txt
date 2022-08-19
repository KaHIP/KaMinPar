list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(TestHelper)

# Copy test instances to binary directory
file(COPY test_instances DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Common -> Datastructures
kaminpar_add_common_test(test_common_preallocated_vector common/datastructures/preallocated_vector_test.cc)
kaminpar_add_common_test(test_common_binary_heap common/datastructures/binary_heap_test.cc)
kaminpar_add_common_test(test_common_fast_reset_array common/datastructures/fast_reset_array_test.cc)
kaminpar_add_common_test(test_common_marker common/datastructures/marker_test.cc)
kaminpar_add_common_test(test_common_static_array common/datastructures/static_array_test.cc)

