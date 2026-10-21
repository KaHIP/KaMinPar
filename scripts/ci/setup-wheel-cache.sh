#!/usr/bin/env bash
set -euo pipefail

# Use portable upstream binaries so manylinux2014 does not need EPEL or a newer libc.
version=0.12.0
case "${RUNNER_OS}:${RUNNER_ARCH}" in
  Linux:X64)
    target=x86_64-unknown-linux-musl
    checksum=b0e89ead6899224a4ba2b90e9073bf1ce036d95bab30f3dc33c1e1468bc4ad44
    ;;
  Linux:ARM64)
    target=aarch64-unknown-linux-musl
    checksum=111ddd28fb108cb3e17edb69ab62cefe1dcc97b02e5006ff9c1330f4f2e78368
    ;;
  macOS:ARM64)
    target=aarch64-apple-darwin
    checksum=4d5281f8760963347b29b9ca4ab1dbde99712c17329619fc9cecba1577ccc8d2
    ;;
  *)
    echo "Unsupported wheel cache platform: ${RUNNER_OS}:${RUNNER_ARCH}" >&2
    exit 1
    ;;
esac

archive="${RUNNER_TEMP}/sccache.tar.gz"
binary_dir="${RUNNER_TEMP}/sccache-bin"
mkdir -p "$binary_dir" "${RUNNER_TEMP}/sccache"
curl --fail --location --retry 5 --retry-delay 2 \
  "https://github.com/mozilla/sccache/releases/download/v${version}/sccache-v${version}-${target}.tar.gz" \
  --output "$archive"
printf '%s  %s\n' "$checksum" "$archive" | shasum -a 256 --check
tar -xzf "$archive" --strip-components=1 -C "$binary_dir" "sccache-v${version}-${target}/sccache"
"${binary_dir}/sccache" --version
