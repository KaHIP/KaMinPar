#!/bin/bash
BUILD_DIR="build"
if [[ $(hostname) == i10pc* ]]; then 
    HOST=$(hostname)
    ID=${HOST#i10pc}
    BUILD_DIR="${BUILD_DIR}${ID}"
fi

OPT=${1:-Dbg}

function get_num_cores {
  case "$(uname)" in
  Darwin)
    sysctl -n hw.ncpu
    ;;
  *)
    grep -c ^processor /proc/cpuinfo
    ;;
  esac
}

# Only use detected core count if CMAKE_BUILD_PARALLEL_LEVEL is not set 
if [[ -z "$CMAKE_BUILD_PARALLEL_LEVEL" ]]; then 
    NCORES=$(get_num_cores)
else 
    NCORES="$CMAKE_BUILD_PARALLEL_LEVEL"
fi 

function build_target {
  cmake --build "$BUILD_DIR" --parallel "$NCORES" --target "$1"
}

# Update submodules 
git submodule update --init --recursive

case $OPT in
    DbgHeavy)
        CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug -DKAMINPAR_ASSERTION_LEVEL=heavy -DKAMINPAR_ENABLE_SANITIZER=On"
        ;;
    Dbg)
        CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Relese -DKAMINPAR_ASSERTION_LEVEL=normal -DKAMINPAR_ENABLE_SANITIZER=On"
        ;;
    DbgLight)
        CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DKAMINPAR_ASSERTION_LEVEL=light"
        ;;
    Rel)
        CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DKAMINPAR_ASSERTION_LEVEL=none"
        ;;
    *)
        echo "Invalid build type option: $OPT"
        echo "Select one of: DbgHeavy Dbg DbgLight Rel"
        exit 1
        ;;
esac

CMAKE_ARGS="$CMAKE_ARGS -DKAMINPAR_ENABLE_GRAPHGEN=On"

# Run CMake
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS

# Build targets
build_target "KaMinPar" # binary
build_target "kaminpar" # library
if cmake --build "$BUILD_DIR" --target help | grep dKaMinPar; then
    build_target "dKaMinPar" # distributed binary
fi

