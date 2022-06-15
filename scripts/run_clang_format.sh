#!/bin/bash

if [[ "$PWD" == */scripts ]]; then
    echo "Script must be run from the project's root directory."
    exit 1
fi

for directory in "apps" "common" "kaminpar" "dkaminpar" "tests" "dtests" "library"; do
    find "$directory"                           \
        -type f                                 \
        \( -name "*.cc" -or -name "*.h" \)   \
        -exec clang-format -i {} \;
done
