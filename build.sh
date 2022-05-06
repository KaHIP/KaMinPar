#!/bin/bash
BUILD_DIR="build"

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

# Run CMake
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release $ADDITIONAL_COMMANDS

# Build targets
build_target "KaMinPar" # binary
build_target "kaminpar" # library
if cmake --build "$BUILD_DIR" --target help | grep dKaMinPar; then
    build_target "dKaMinPar" # distributed binary
fi

