#!/bin/bash
cd "${0%/}" || exit # run from source directory or exit

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
  cmake --build build --parallel "$NCORES" --target $1
}

git submodule update --init --recursive

PROJECT_ROOT=$(pwd)
if [ ! -f build/Makefile ]; then
  ADDITIONAL_COMMANDS=""

  mkdir -p build &&
    cd build &&
    cmake .. -DCMAKE_BUILD_TYPE=Release $ADDITIONAL_COMMANDS
  cd "$PROJECT_ROOT" || exit
fi
if [ ! -f build/Makefile ]; then
  echo "Unable to create Makefile in build/"
  exit
fi

build_target "KaMinPar" # binary
build_target "kaminpar" # library
build_target "GraphChecker" # tools
build_target "GraphConverter" # tools
build_target "VerifyPartition" # tools
if [[ $1 == "DISTRIBUTED" ]]; then build_target "dKaMinPar"; fi # distributed binary
