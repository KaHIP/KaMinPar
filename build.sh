#!/bin/bash
git submodule update --init

PROJECT_ROOT=$(pwd)
if [ ! -f build/Makefile ]; then
  mkdir -p build &&
    cd build &&
    cmake .. -DCMAKE_BUILD_TYPE=Release
  cd "$PROJECT_ROOT" || exit
fi
if [ ! -f build/Makefile ]; then
  echo "Unable to create Makefile in build/"
  exit
fi
cmake --build build --target KaMinPar
