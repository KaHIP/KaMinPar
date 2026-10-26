# Add a scheduling hook to a test-private copy of the real vendored strategy.
# The production header and all of its operations remain otherwise unchanged.
if(NOT DEFINED SOURCE_HEADER OR NOT DEFINED OUTPUT_HEADER)
    message(FATAL_ERROR "SOURCE_HEADER and OUTPUT_HEADER are required")
endif()

file(READ "${SOURCE_HEADER}" source)
set(protect "_table = _rec_handle.protect(_global._table);")
string(FIND "${source}" "${protect}" position)
if(position EQUAL -1)
    message(FATAL_ERROR "Growt snapshot acquisition changed; update the regression hook")
endif()
string(REPLACE "${protect}"
    "${protect}\n    ::growt_snapshot_test::after_protect();"
    instrumented "${source}")
get_filename_component(output_directory "${OUTPUT_HEADER}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
file(WRITE "${OUTPUT_HEADER}" "${instrumented}")
