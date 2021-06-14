#!/bin/bash
function get_num_cores {
  if [[ $(uname) == "Linux" ]]; then grep -c ^processor /proc/cpuinfo; fi
  if [[ $(uname) == "Darwin" ]]; then sysctl -n hw.ncpu; fi
}

git submodule update --init

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
cmake --build build --parallel "$(get_num_cores)" --target KaMinPar
