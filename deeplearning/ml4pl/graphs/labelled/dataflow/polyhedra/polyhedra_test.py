"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/polyhedra."""
import typing

import networkx as nx
import numpy as np
import pytest

from compilers.llvm import clang
from compilers.llvm import opt
from deeplearning.ml4pl.graphs.labelled.dataflow.polyhedra import polyhedra
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


class InputPair(typing.NamedTuple):
  # A <graph, bytecode> tuple for inputting to the polyhedral detection code.
  graph: nx.MultiDiGraph
  bytecode: str


def CSourceToBytecode(source: str) -> str:
  """Build LLVM bytecode for the given C code."""
  # NOTE: This has to be at least -O1 to obtain polly outputs
  process = clang.Exec(
    ["-xc", "-O1", "-S", "-emit-llvm", "-", "-o", "-"], stdin=source
  )
  assert not process.returncode
  return process.stdout


def CSourceToInput(source: str) -> str:
  """Create a bytecode for the given C source string.
  This is a convenience method for generating test inputs. If this method fails,
  it is because clang is broken.
  """
  bytecode = CSourceToBytecode(source)
  return bytecode


def test_MakePolyhedralGraphs_invalid_bytecode():
  graph = nx.MultiDiGraph()
  bytecode = "invalid bytecode!"
  with test.Raises(opt.OptException):
    list(polyhedra.MakePolyhedralGraphs(bytecode))


def test_MakePolyhedralGraphs_basic_gemm():
  # Snippet adapted from Polybench 4.2, gemm.c
  bytecode = CSourceToInput(
    """
void A(double alpha, double beta, double C[1000][1100], double A[1000][1200], double B[1200][1100]) {
  int i, j, k;
  for (i = 0; i < 1000; i++) {
    for (j = 0; j < 1100; j++)
        C[i][j] *= beta;
    for (k = 0; k < 1200; k++) {
       for (j = 0; j < 1100; j++)
          C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
}
"""
  )
  graphs = list(polyhedra.MakePolyhedralGraphs(bytecode))
  assert len(graphs) == 1

  # Computed by polly separately.
  polyhedral_identifiers = [f"%{i}" for i in range(6, 37)]

  for node, data in graphs[0].nodes(data=True):
    # Test the 'selector' node.
    assert data["x"][1] == 0
    # Test the labels.
    if any(
      "original_text" in data and data["original_text"].startswith(identifier)
      for identifier in polyhedral_identifiers
    ):
      if data["y"][1] != 1:
        raise ValueError("Identifier is not polyhedral: " + str(data))

      assert np.array_equal(data["y"], [0, 1])


if __name__ == "__main__":
  test.Main()
