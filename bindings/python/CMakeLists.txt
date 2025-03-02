cmake_minimum_required(VERSION 3.21)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  DESCRIPTION "Python Bindings for KaMinPar"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(KAMINPAR_PYTHON_64BIT "Build with 64-bit IDs" OFF)
option(KAMINPAR_PYTHON_INSTALL_TBB "Install TBB with the package" OFF)

message(STATUS "Building with 64-bit IDs: ${KAMINPAR_PYTHON_64BIT}")
message(STATUS "Installing TBB with the package: ${KAMINPAR_PYTHON_INSTALL_TBB}")

include(FetchContent)

# TODO: Update when the branch is merged
FetchContent_Declare(
  KaMinPar
  GIT_REPOSITORY https://github.com/dsalwasser/KaMinPar.git
  GIT_TAG shm/feat/graph-compression-interface
  EXCLUDE_FROM_ALL
)
set(KAMINPAR_DOWNLOAD_TBB ${KAMINPAR_PYTHON_INSTALL_TBB} CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(KaMinPar)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(kaminpar_python src/PyKaMinPar.cpp)
target_compile_definitions(kaminpar_python PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(kaminpar_python PUBLIC KaMinPar::KaMinPar KaMinPar::KaMinParIO)

add_library(KaMinPar::PyKaMinPar ALIAS kaminpar_python)
install(TARGETS kaminpar_python DESTINATION kaminpar)

if(KAMINPAR_PYTHON_INSTALL_TBB)
  install(TARGETS tbb tbbmalloc DESTINATION /tmp/tbb)
endif()
