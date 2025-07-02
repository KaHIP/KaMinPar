#!/usr/bin/env bash

if [[ "$PWD" == */scripts ]]; then
    echo "Script must be run from the project's root directory."
    exit 1
fi

from=${1:-}
to=${2:-}

IFS='.' read -r from_major from_minor from_patch <<< "$from"
IFS='.' read -r to_major to_minor to_patch <<< "$to"

sed -i "s/#define KAMINPAR_VERSION_MAJOR $from_major/#define KAMINPAR_VERSION_MAJOR $to_major/g" include/kaminpar-shm/kaminpar.h
sed -i "s/#define KAMINPAR_VERSION_MINOR $from_minor/#define KAMINPAR_VERSION_MINOR $to_minor/g" include/kaminpar-shm/kaminpar.h
sed -i "s/#define KAMINPAR_VERSION_PATCH $from_patch/#define KAMINPAR_VERSION_PATCH $to_patch/g" include/kaminpar-shm/kaminpar.h

sed -i "s/#define CKAMINPAR_VERSION_MAJOR $from_major/#define CKAMINPAR_VERSION_MAJOR $to_major/g" include/kaminpar-shm/ckaminpar.h
sed -i "s/#define CKAMINPAR_VERSION_MINOR $from_minor/#define CKAMINPAR_VERSION_MINOR $to_minor/g" include/kaminpar-shm/ckaminpar.h
sed -i "s/#define CKAMINPAR_VERSION_PATCH $from_patch/#define CKAMINPAR_VERSION_PATCH $to_patch/g" include/kaminpar-shm/ckaminpar.h

sed -i "s/VERSION $from/VERSION $to/g" CMakeLists.txt
sed -i "s/version = \"$from\"/version = \"$to\"/g" flake.nix
sed -i "s/$from/$to/g" README.MD
sed -i "s/version = \"$from\"/version = \"$to\"/g" bindings/networkit/pyproject.toml
sed -i "s/version = \"$from\"/version = \"$to\"/g" bindings/python/pyproject.toml
sed -i "s/__version__ == \"$from\"/__version__ == \"$to\"/g" bindings/python/tests/test_pykaminpar.py
