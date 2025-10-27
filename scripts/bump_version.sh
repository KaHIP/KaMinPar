#!/usr/bin/env bash

if [[ "$PWD" == */scripts ]]; then
    echo "Script must be run from the project's root directory."
    exit 1
fi
if [[ "$#" -lt 2 ]]; then
    echo "Usage: $0 x.y.z X.Y.Z"
    exit 1
fi

from="$1"
IFS='.' read -r from_major from_minor from_patch <<< "$from"

to="$2"
IFS='.' read -r to_major to_minor to_patch <<< "$to"

echo "Bumping version from $from_major-$from_minor-$from_patch to $to_major-$to_minor-$to_patch"

if sed --version >/dev/null 2>&1; then
    # GNU sed
    SED_INPLACE=(-i)
else
    # macOS sed
    SED_INPLACE=(-i '')
fi

sed "${SED_INPLACE[@]}" "s/#define KAMINPAR_VERSION_MAJOR $from_major/#define KAMINPAR_VERSION_MAJOR $to_major/g" include/kaminpar-shm/kaminpar.h
sed "${SED_INPLACE[@]}" "s/#define KAMINPAR_VERSION_MINOR $from_minor/#define KAMINPAR_VERSION_MINOR $to_minor/g" include/kaminpar-shm/kaminpar.h
sed "${SED_INPLACE[@]}" "s/#define KAMINPAR_VERSION_PATCH $from_patch/#define KAMINPAR_VERSION_PATCH $to_patch/g" include/kaminpar-shm/kaminpar.h

sed "${SED_INPLACE[@]}" "s/#define CKAMINPAR_VERSION_MAJOR $from_major/#define CKAMINPAR_VERSION_MAJOR $to_major/g" include/kaminpar-shm/ckaminpar.h
sed "${SED_INPLACE[@]}" "s/#define CKAMINPAR_VERSION_MINOR $from_minor/#define CKAMINPAR_VERSION_MINOR $to_minor/g" include/kaminpar-shm/ckaminpar.h
sed "${SED_INPLACE[@]}" "s/#define CKAMINPAR_VERSION_PATCH $from_patch/#define CKAMINPAR_VERSION_PATCH $to_patch/g" include/kaminpar-shm/ckaminpar.h

sed "${SED_INPLACE[@]}" "s/VERSION $from/VERSION $to/g" CMakeLists.txt
sed "${SED_INPLACE[@]}" "s/version = \"$from\"/version = \"$to\"/g" flake.nix
sed "${SED_INPLACE[@]}" "s/$from/$to/g" README.MD
sed "${SED_INPLACE[@]}" "s/version = \"$from\"/version = \"$to\"/g" bindings/networkit/pyproject.toml
sed "${SED_INPLACE[@]}" "s/version = \"$from\"/version = \"$to\"/g" bindings/python/pyproject.toml
sed "${SED_INPLACE[@]}" "s/__version__ == \"$from\"/__version__ == \"$to\"/g" bindings/python/tests/test_pykaminpar.py
