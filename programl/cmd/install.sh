#!/usr/bin/env bash
#
# Build an optimized binary and copy it, along with all of its runfiles, to the
# target prefix.

set -eu

PREFIX=/usr/local/opt/programl

mkdir -pv "$PREFIX/bin"

EXES=(
  graph2cdfg
  graph2dot
  graph2json
  graph2nx
  inst2vec
  llvm2graph
  node_text_freq
  pb2pbtxt
  select_vocab
  xla2graph
)

install() {
  for exe in "${EXES[@]}"; do
    rm -f "$PREFIX/bin/$exe"
    cp -v programl/cmd/$exe "$PREFIX/bin/$exe"
  done
}
install
