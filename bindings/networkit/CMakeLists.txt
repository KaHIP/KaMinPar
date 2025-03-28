cmake_minimum_required(VERSION 3.21)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  DESCRIPTION "NetworKit Bindings for KaMinPar"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(KAMINPAR_PYTHON_INSTALL_TBB "Install TBB with the package" OFF)
message(STATUS "Installing TBB with the package: ${KAMINPAR_PYTHON_INSTALL_TBB}")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Cython MODULE REQUIRED VERSION 3.0)

# To link against the NetworKit library, we need to find its location on the system. We do
# this by running a Python script that imports the NetworKit module and prints its directory.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import networkit, os; print(os.path.dirname(networkit.__file__))"
  OUTPUT_VARIABLE NETWORKIT_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(NETWORKIT_LIB "${NETWORKIT_DIR}/../libnetworkit.so")
set(NETWORKIT_INCLUDE "${NETWORKIT_DIR}/../include")

include(FetchContent)
FetchContent_Declare(
  KaMinPar
  GIT_REPOSITORY https://github.com/KaHIP/KaMinPar.git
  GIT_TAG main
  EXCLUDE_FROM_ALL
)
set(KAMINPAR_DOWNLOAD_TBB ${KAMINPAR_PYTHON_INSTALL_TBB} CACHE BOOL "" FORCE)
set(KAMINPAR_BUILD_WITH_MTUNE_NATIVE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(KaMinPar)

include(UseCython)
cython_transpile(src/kaminpar_networkit/nkkaminpar.pyx
  LANGUAGE CXX
  OUTPUT_VARIABLE nkkaminpar_cpp
)

python_add_library(nkkaminpar MODULE "${nkkaminpar_cpp}" src/kaminpar_networkit.h src/kaminpar_networkit.cc WITH_SOABI)
target_link_libraries(nkkaminpar PUBLIC KaMinPar::KaMinParFull "${NETWORKIT_LIB}")
target_include_directories(nkkaminpar PUBLIC "${NETWORKIT_INCLUDE}" "${NETWORKIT_DIR}" "src/")

add_library(KaMinPar::NetworKitKaMinPar ALIAS nkkaminpar)
install(TARGETS nkkaminpar DESTINATION kaminpar_networkit)

if(KAMINPAR_PYTHON_INSTALL_TBB)
  install(TARGETS tbb tbbmalloc DESTINATION /tmp/tbb)
endif()
