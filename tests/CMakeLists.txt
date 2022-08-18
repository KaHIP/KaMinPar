list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(TestHelper)

# Copy test instances to binary directory
file(COPY test_instances DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Common -> Datastructures
kaminpar_add_common_test(test_common_utils_preallocated_vector common/preallocated_vector_test.cc)
