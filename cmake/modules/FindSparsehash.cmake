find_package(PkgConfig)
pkg_check_modules(PC_Sparsehash QUIET libsparsehash)

find_path(Sparsehash_INCLUDE_DIR
  NAMES google/sparsehash/sparsehashtable.h
  PATHS ${PC_Sparsehash_INCLUDE_DIRS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sparsehash
  FOUND_VAR Sparsehash_FOUND
  REQUIRED_VARS
  Sparsehash_INCLUDE_DIR
)

if (Sparsehash_FOUND)
  message(STATUS "Found libsparsehash: ${Sparsehash_INCLUDE_DIR}")
  set(Sparsehash_INCLUDE_DIRS ${Sparsehash_INCLUDE_DIR})
endif()

if(Sparsehash_FOUND AND NOT TARGET Sparsehash::Sparsehash)
  add_library(Sparsehash::Sparsehash INTERFACE IMPORTED)
  target_include_directories(Sparsehash::Sparsehash SYSTEM INTERFACE "${Sparsehash_INCLUDE_DIRS}")
endif()


mark_as_advanced(
  Sparsehash_INCLUDE_DIR
)
