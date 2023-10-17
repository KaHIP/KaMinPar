#!/bin/bash

if [[ "$PWD" == */scripts ]]; then
    echo "Script must be run from the project's root directory."
    exit 1
fi

for directory in "apps" \
    "tests" \
    "kaminpar-common" \
    "kaminpar-shm" \
    "kaminpar-dist" \
    "kaminpar-cli" \
    "kaminpar-mpi"; do
    find "$directory"                        \
        -type f                              \
        \( -name "*.cc" -or -name "*.h" \)   \
        -exec clang-format -i {} \;
done
