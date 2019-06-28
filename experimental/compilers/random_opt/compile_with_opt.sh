#!/usr/bin/env bash
set -eux

# The URL to grab a source tarball from.
export URL="https://github.com/enthought/bzip2-1.0.6/archive/master.tar.gz"
# The list of basenames for all sources.
export SOURCES="blocksort.c bzlib.c compress.c crctable.c decompress.c huffman.c randtable.c bzip2.c"
# The working directory for this script.
export WORKDIR="/tmp/phd/experimental/compilers/random_opt/compile_with_opt"

main() {
  mkdir -p "$WORKDIR/src" "$WORKDIR/bc"
  wget "$URL" -O "$WORKDIR/srcs.tar.gz"
  tar --strip-components=1 -xzvf "$WORKDIR/srcs.tar.gz" -C "$WORKDIR/src"

  # Check that all the expected files exist.
  for src in $SOURCES; do
    test -f "$WORKDIR/src/$src"
  done

  # Compile bitcode for all sources.
  for src in $SOURCES; do
    bazel run //compilers/llvm:clang -- -- \
        "$WORKDIR/src/$src" -o "$WORKDIR/bc/$src" -S -emit-llvm -c -O0 -g
  done

  BITCODES=""
  for src in $SOURCES; do
    BITCODES="$WORKDIR/bc/$src $BITCODES"
  done

  # Link bitcode files into a single bytecode file.
  bazel run //compilers/llvm:llvm_link -- -- \
    $BITCODES -o "$WORKDIR/source.ll" -S

  bazel run //compilers/llvm:opt -- -- \
    -O3 "$WORKDIR/source.ll" -S -o "$WORKDIR/source_optimized.ll"

  bazel run //compilers/llvm:clang -- -- \
    -O3 "$WORKDIR/source.ll" -S -emit-llvm -o "$WORKDIR/source_optimized2.ll"

  # Compile unoptimized binary from bytecode.
  bazel run //compilers/llvm:clang -- -- \
    -O0 "$WORKDIR/source.ll" -o "$WORKDIR/exe_O0"
  # Compile optimized binary from bytecode.
  bazel run //compilers/llvm:clang -- -- \
    -O3 "$WORKDIR/source.ll" -o "$WORKDIR/exe_O3"
}
main $@
