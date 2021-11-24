#!/bin/bash
BUILD_DIR="build"

#cd "${0%/}" || exit # run from source directory or exit

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
NCORES=$(get_num_cores)
[ -n "$NCORES" ] || NCORES=4

function build_target {
  cmake --build "$BUILD_DIR" --parallel "$NCORES" --target $1
}

git submodule update --init --recursive

PROJECT_ROOT=$(pwd)
if [ ! -f build/Makefile ]; then
  ADDITIONAL_COMMANDS=""

  mkdir -p "$BUILD_DIR" &&
    cd "$BUILD_DIR" &&
    cmake .. -DCMAKE_BUILD_TYPE=Release $ADDITIONAL_COMMANDS
  cd "$PROJECT_ROOT" || exit
fi
if [ ! -f build/Makefile ]; then
  echo "Unable to create Makefile in build/"
  exit
fi

build_target "KaMinPar" # binary
build_target "kaminpar" # library

if cmake --build "$BUILD_DIR" --target help | grep dKaMinPar; then
    build_target "dKaMinPar"
fi

