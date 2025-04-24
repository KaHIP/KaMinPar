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

option(KAMINPAR_PYTHON_64BIT "Build with 64-bit IDs" ON)
option(KAMINPAR_PYTHON_INSTALL_TBB "Install TBB with the package" OFF)

message(STATUS "Building with 64-bit IDs: ${KAMINPAR_PYTHON_64BIT}")
message(STATUS "Installing TBB with the package: ${KAMINPAR_PYTHON_INSTALL_TBB}")

include(FetchContent)
FetchContent_Declare(
  KaMinPar
  GIT_REPOSITORY https://github.com/KaHIP/KaMinPar.git
  GIT_TAG main
  EXCLUDE_FROM_ALL
)
set(KAMINPAR_64BIT_IDS ${KAMINPAR_PYTHON_64BIT} CACHE BOOL "" FORCE)
set(KAMINPAR_64BIT_WEIGHTS ${KAMINPAR_PYTHON_64BIT} CACHE BOOL "" FORCE)
set(KAMINPAR_DOWNLOAD_TBB ${KAMINPAR_PYTHON_INSTALL_TBB} CACHE BOOL "" FORCE)
set(KAMINPAR_BUILD_WITH_MTUNE_NATIVE OFF CACHE BOOL "" FORCE)
set(KAMINPAR_BUILD_WITH_SPARSEHASH OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(KaMinPar)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(kaminpar_python src/PyKaMinPar.cpp)
target_link_libraries(kaminpar_python PUBLIC KaMinPar::KaMinParFull)
target_compile_definitions(kaminpar_python PRIVATE VERSION_INFO=${PROJECT_VERSION})
if(KAMINPAR_PYTHON_64BIT)
  target_compile_definitions(kaminpar_python PRIVATE KAMINPAR_PYTHON_64BIT)
endif()

add_library(KaMinPar::PyKaMinPar ALIAS kaminpar_python)
install(TARGETS kaminpar_python DESTINATION kaminpar)

if(KAMINPAR_PYTHON_INSTALL_TBB)
  install(TARGETS tbb tbbmalloc DESTINATION /tmp/tbb)
endif()
