# Copyright 2019-2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //compilers/llvm/clang.py."""
import pathlib

import pytest

from compilers.llvm import clang
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import fs
from labm8.py import test

FLAGS = app.FLAGS

_BASIC_CPP_PROGRAM = """
int main() {
  return 0;
}
"""

# The C source code for an "n queens" puzzle.
_NQUEENS_SRC = fs.Read(bazelutil.DataPath("phd/datasets/benchmarks/nqueens.cc"))


def _StripPreprocessorLines(out: str):
  return "\n".join(line for line in out.split("\n") if not line.startswith("#"))


# Exec() tests.


def test_Exec_compile_bytecode(tempdir: pathlib.Path):
  """Test bytecode generation."""
  with open(tempdir / "foo.cc", "w") as f:
    f.write(_BASIC_CPP_PROGRAM)
  p = clang.Exec(
    [
      str(tempdir / "foo.cc"),
      "-xc++",
      "-S",
      "-emit-llvm",
      "-c",
      "-o",
      str(tempdir / "foo.ll"),
    ]
  )
  assert not p.stderr
  assert not p.stdout
  assert not p.returncode
  assert (tempdir / "foo.ll").is_file()


def test_Exec_compile_bytecode_stdin(tempdir: pathlib.Path):
  """Test bytecode generation."""
  p = clang.Exec(
    ["-xc++", "-S", "-emit-llvm", "-c", "-o", str(tempdir / "foo.ll"), "-"],
    stdin=_BASIC_CPP_PROGRAM,
  )
  print(p.stderr)
  assert not p.stderr
  assert not p.stdout
  assert not p.returncode
  assert (tempdir / "foo.ll").is_file()


@test.Parametrize("opt", ("-O0", "-O1", "-O2", "-O3", "-Ofast", "-Os", "-Oz"))
def test_ValidateOptimizationLevel_valid(opt: str):
  """Test that valid optimization levels are returned."""
  assert clang.ValidateOptimizationLevel(opt) == opt


@test.Parametrize(
  "opt", ("O0", "-O4", "foo")  # missing leading '-'  # not a real value
)  # not a real value
def test_ValidateOptimizationLevel_invalid(opt: str):
  """Test that invalid optimization levels raise an error."""
  with test.Raises(ValueError) as e_ctx:
    clang.ValidateOptimizationLevel(opt)
  assert opt in str(e_ctx.value)


# Preprocess() tests.


def test_Preprocess_empty_input():
  """Test that Preprocess accepts an empty input."""
  assert _StripPreprocessorLines(clang.Preprocess("")) == "\n"


def test_Preprocess_small_cxx_program():
  """Test pre-processing a small C++ program."""
  assert clang.Preprocess(
    """
#define FOO T
template<typename FOO>
FOO foobar(const T& a) {return a;}

int foo() { return foobar<int>(10); }
""",
    copts=["-xc++"],
  ).endswith(
    """

template<typename T>
T foobar(const T& a) {return a;}

int foo() { return foobar<int>(10); }
"""
  )


def test_Preprocess_missing_include():
  """Test that Preprocessor error is raised on missing #include."""
  with test.Raises(clang.ClangException) as e_info:
    clang.Preprocess('#include "my-missing-file.h"')
  assert "'my-missing-file.h' file not found" in str(e_info.value.stderr)


def test_ClangBisectMessageToInvocation_inalid():
  with test.Raises(clang.ClangException) as e_ctx:
    clang.ClangBisectMessageToInvocation("foo")
  assert "Cannot interpret line: foo" == str(e_ctx.value.msg)


def test_ClangBisectMessageToInvocation_valid():
  invocation = clang.ClangBisectMessageToInvocation(
    "BISECT: running pass (1) Simplify the CFG on function "
    "(_Z16show_short_boardP7NQueensPi)"
  )
  assert invocation.name == "Simplify the CFG"
  assert invocation.target_type == "function"
  assert invocation.target == "_Z16show_short_boardP7NQueensPi"


@test.Parametrize(
  "optimization_level", ("-O0", "-O1", "-O2", "-O3", "-Ofast", "-Os", "-Oz")
)
@test.LinuxTest()
def test_GetOptPasses_O3_language_equivalence(optimization_level: str):
  """Test that C/C++ passes run are the same."""
  c_args = clang.GetOptPasses([optimization_level], language="c")
  cxx_args = clang.GetOptPasses([optimization_level], language="c++")
  assert c_args == cxx_args


@test.LinuxTest()
def test_GetOptPasses_O3_cxx():
  """Test optimisations ran at -O3 for an empty C++ file."""
  args = clang.GetOptPasses(["-O3"], language="c++")
  assert args == [
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="SROA", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Early CSE", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Infer set function attributes", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Call-site splitting", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Interprocedural Sparse Conditional Constant Propagation",
      target="-",
      target_type="module",
    ),
    clang.OptPassRunInvocation(
      name="Called Value Propagation", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Global Variable Optimizer", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Promote Memory to Register", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Dead Argument Elimination", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="main",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining", target="main", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes", target="main", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="main",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="SROA", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Early CSE w/ MemorySSA", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Speculatively execute instructions if target has divergent branches",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Call Elimination", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Reassociate expressions", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="MergedLoadStoreMotion", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Global Value Numbering", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="MemCpy Optimization", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Sparse Conditional Constant Propagation",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Bit-Tracking Dead Code Elimination",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Dead Store Elimination", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Aggressive Dead Code Elimination",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Eliminate Available Externally Globals",
      target="-",
      target_type="module",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes in RPO", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Global Variable Optimizer", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Dead Global Elimination", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Float to int", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Loop Distribution", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Loop Vectorization", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Loop Load Elimination", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="SLP Vectorizer", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Alignment from assumptions", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Strip Unused Function Prototypes", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Dead Global Elimination", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Merge Duplicate Global Constants", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Remove redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Hoist/decompose integer division and remainder",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Expand memcmp() to load/stores",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Constant Hoisting", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Partially inline calls to library functions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Scalarize Masked Memory Intrinsics",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="CodeGen Prepare", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 DAG->DAG Instruction Selection",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Local Dynamic TLS Access Clean-up",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Domain Reassignment Pass", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Optimize machine instruction PHIs",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Early If-Conversion", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 cmov Conversion", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Common Subexpression Elimination",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine code sinking", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Peephole Optimizations", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Live Range Shrink", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Optimize", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 Optimize Call Frame", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Two-Address instruction pass", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Machine Instruction Scheduler",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Shrink Wrapping analysis", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Control Flow Optimizer", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Machine Copy Propagation Pass",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Post RA top-down list latency scheduler",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Branch Probability Basic Block Placement",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Execution Dependency Fix", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 Byte/Word Instruction Fixup",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Atom pad short functions", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Fixup", target="main", target_type="function"
    ),
  ]


@test.LinuxTest()
def test_GetOptPasses_O3_nqueens():
  """Black box opt passes test for -O0."""
  args = clang.GetOptPasses(["-O3"], stubfile=_NQUEENS_SRC, language="c++")
  assert args == [
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="SROA",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Early CSE",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="SROA",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Early CSE",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="SROA", target="_Z11check_placePiii", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Early CSE", target="_Z11check_placePiii", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="SROA", target="_Z9put_queenP7NQueensPii", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Early CSE",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="SROA", target="_Z5solveP7NQueens", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Early CSE", target="_Z5solveP7NQueens", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="SROA", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Early CSE", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Infer set function attributes", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Call-site splitting",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Call-site splitting",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Call-site splitting",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Call-site splitting",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Call-site splitting",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Call-site splitting", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Interprocedural Sparse Conditional Constant Propagation",
      target="-",
      target_type="module",
    ),
    clang.OptPassRunInvocation(
      name="Called Value Propagation", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Global Variable Optimizer", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Promote Memory to Register",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Promote Memory to Register",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Promote Memory to Register",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Promote Memory to Register",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Promote Memory to Register",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Promote Memory to Register", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Dead Argument Elimination", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="putchar",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining", target="putchar", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes", target="putchar", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="putchar",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="printf",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining", target="printf", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes", target="printf", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="printf",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="SROA",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Early CSE w/ MemorySSA",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Speculatively execute instructions if target has divergent branches",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Call Elimination",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Reassociate expressions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unswitch loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Induction Variable Simplification", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Recognize loop idioms", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Delete dead loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="MergedLoadStoreMotion",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Global Value Numbering",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="MemCpy Optimization",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Sparse Conditional Constant Propagation",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Bit-Tracking Dead Code Elimination",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Dead Store Elimination",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Aggressive Dead Code Elimination",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="llvm.lifetime.start.p0i8",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="llvm.lifetime.start.p0i8",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="llvm.lifetime.start.p0i8",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="llvm.lifetime.start.p0i8",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="llvm.lifetime.end.p0i8",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="llvm.lifetime.end.p0i8",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="llvm.lifetime.end.p0i8",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="llvm.lifetime.end.p0i8",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="SROA",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Early CSE w/ MemorySSA",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Speculatively execute instructions if target has divergent branches",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Call Elimination",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Reassociate expressions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unswitch loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unswitch loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Induction Variable Simplification", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Recognize loop idioms", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Delete dead loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Induction Variable Simplification", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Recognize loop idioms", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Delete dead loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="MergedLoadStoreMotion",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Global Value Numbering",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="MemCpy Optimization",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Sparse Conditional Constant Propagation",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Bit-Tracking Dead Code Elimination",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Dead Store Elimination",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Aggressive Dead Code Elimination",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="_Z11check_placePiii",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="_Z11check_placePiii",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="_Z11check_placePiii",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="_Z11check_placePiii",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="SROA", target="_Z11check_placePiii", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Early CSE w/ MemorySSA",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Speculatively execute instructions if target has divergent branches",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Call Elimination",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Reassociate expressions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unswitch loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Induction Variable Simplification", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Recognize loop idioms", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Delete dead loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="MergedLoadStoreMotion",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Global Value Numbering",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="MemCpy Optimization",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Sparse Conditional Constant Propagation",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Bit-Tracking Dead Code Elimination",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Dead Store Elimination",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Aggressive Dead Code Elimination",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="_Z9put_queenP7NQueensPii",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="_Z9put_queenP7NQueensPii",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="_Z9put_queenP7NQueensPii",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="_Z9put_queenP7NQueensPii",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="SROA", target="_Z9put_queenP7NQueensPii", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Early CSE w/ MemorySSA",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Speculatively execute instructions if target has divergent branches",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Call Elimination",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Reassociate expressions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unswitch loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unswitch loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unswitch loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unswitch loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unswitch loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unswitch loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Induction Variable Simplification", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Recognize loop idioms", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Delete dead loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Induction Variable Simplification", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Recognize loop idioms", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Delete dead loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Induction Variable Simplification", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Recognize loop idioms", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Delete dead loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Induction Variable Simplification", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Recognize loop idioms", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Delete dead loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Induction Variable Simplification", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Recognize loop idioms", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Delete dead loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="MergedLoadStoreMotion",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Global Value Numbering",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="MemCpy Optimization",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Sparse Conditional Constant Propagation",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Bit-Tracking Dead Code Elimination",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Dead Store Elimination",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Aggressive Dead Code Elimination",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="malloc",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining", target="malloc", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes", target="malloc", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="malloc",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="free",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining", target="free", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes", target="free", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="free",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="_Z5solveP7NQueens",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="_Z5solveP7NQueens",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="_Z5solveP7NQueens",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="_Z5solveP7NQueens",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="SROA", target="_Z5solveP7NQueens", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Early CSE w/ MemorySSA",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Speculatively execute instructions if target has divergent branches",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading", target="_Z5solveP7NQueens", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Call Elimination",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Reassociate expressions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="MergedLoadStoreMotion",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Global Value Numbering",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="MemCpy Optimization",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Sparse Conditional Constant Propagation",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Bit-Tracking Dead Code Elimination",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading", target="_Z5solveP7NQueens", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Dead Store Elimination",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Aggressive Dead Code Elimination",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="llvm.memset.p0i8.i64",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="llvm.memset.p0i8.i64",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="llvm.memset.p0i8.i64",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="llvm.memset.p0i8.i64",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="main",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining", target="main", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes", target="main", target_type="SCC"
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="main",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="SROA", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Early CSE w/ MemorySSA", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Speculatively execute instructions if target has divergent branches",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Call Elimination", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Reassociate expressions", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="MergedLoadStoreMotion", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Global Value Numbering", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="MemCpy Optimization", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Sparse Conditional Constant Propagation",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Bit-Tracking Dead Code Elimination",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Jump Threading", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Value Propagation", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Dead Store Elimination", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Aggressive Dead Code Elimination",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="llvm.memcpy.p0i8.p0i8.i64",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="llvm.memcpy.p0i8.p0i8.i64",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="llvm.memcpy.p0i8.p0i8.i64",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="llvm.memcpy.p0i8.p0i8.i64",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Remove unused exception handling info",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Function Integration/Inlining",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Promote 'by reference' arguments to scalars",
      target="<<null function>>",
      target_type="SCC",
    ),
    clang.OptPassRunInvocation(
      name="Eliminate Available Externally Globals",
      target="-",
      target_type="module",
    ),
    clang.OptPassRunInvocation(
      name="Deduce function attributes in RPO", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Global Variable Optimizer", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Dead Global Elimination", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Float to int",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Distribution",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Vectorization",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Load Elimination",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="SLP Vectorizer",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Alignment from assumptions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Float to int",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Distribution",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Vectorization",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Load Elimination",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="SLP Vectorizer",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Alignment from assumptions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Float to int", target="_Z11check_placePiii", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Distribution",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Vectorization",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Load Elimination",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="SLP Vectorizer",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Alignment from assumptions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Float to int",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Rotate Loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Distribution",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Vectorization",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Load Elimination",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="SLP Vectorizer",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Unroll loops", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Invariant Code Motion", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Alignment from assumptions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Float to int", target="_Z5solveP7NQueens", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Loop Distribution",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Vectorization",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Load Elimination",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="SLP Vectorizer", target="_Z5solveP7NQueens", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Alignment from assumptions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Float to int", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Loop Distribution", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Loop Vectorization", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Loop Load Elimination", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="SLP Vectorizer", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Combine redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Alignment from assumptions", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Strip Unused Function Prototypes", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Dead Global Elimination", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(
      name="Merge Duplicate Global Constants", target="-", target_type="module"
    ),
    clang.OptPassRunInvocation(name="Loop Sink", target="", target_type="loop"),
    clang.OptPassRunInvocation(
      name="Remove redundant instructions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Hoist/decompose integer division and remainder",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(name="Loop Sink", target="", target_type="loop"),
    clang.OptPassRunInvocation(name="Loop Sink", target="", target_type="loop"),
    clang.OptPassRunInvocation(
      name="Remove redundant instructions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Hoist/decompose integer division and remainder",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(name="Loop Sink", target="", target_type="loop"),
    clang.OptPassRunInvocation(
      name="Remove redundant instructions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Hoist/decompose integer division and remainder",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(name="Loop Sink", target="", target_type="loop"),
    clang.OptPassRunInvocation(name="Loop Sink", target="", target_type="loop"),
    clang.OptPassRunInvocation(name="Loop Sink", target="", target_type="loop"),
    clang.OptPassRunInvocation(name="Loop Sink", target="", target_type="loop"),
    clang.OptPassRunInvocation(name="Loop Sink", target="", target_type="loop"),
    clang.OptPassRunInvocation(
      name="Remove redundant instructions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Hoist/decompose integer division and remainder",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove redundant instructions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Hoist/decompose integer division and remainder",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove redundant instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Hoist/decompose integer division and remainder",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Simplify the CFG", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Loop Strength Reduction", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Expand memcmp() to load/stores",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Constant Hoisting",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Partially inline calls to library functions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Scalarize Masked Memory Intrinsics",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="CodeGen Prepare",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Strength Reduction", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Strength Reduction", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Expand memcmp() to load/stores",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Constant Hoisting",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Partially inline calls to library functions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Scalarize Masked Memory Intrinsics",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="CodeGen Prepare",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Strength Reduction", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Expand memcmp() to load/stores",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Constant Hoisting",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Partially inline calls to library functions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Scalarize Masked Memory Intrinsics",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="CodeGen Prepare",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Loop Strength Reduction", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Strength Reduction", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Strength Reduction", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Strength Reduction", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Loop Strength Reduction", target="", target_type="loop"
    ),
    clang.OptPassRunInvocation(
      name="Expand memcmp() to load/stores",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Constant Hoisting",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Partially inline calls to library functions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Scalarize Masked Memory Intrinsics",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="CodeGen Prepare",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Expand memcmp() to load/stores",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Constant Hoisting",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Partially inline calls to library functions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Scalarize Masked Memory Intrinsics",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="CodeGen Prepare", target="_Z5solveP7NQueens", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Expand memcmp() to load/stores",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Constant Hoisting", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Partially inline calls to library functions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Scalarize Masked Memory Intrinsics",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="CodeGen Prepare", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 DAG->DAG Instruction Selection",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Local Dynamic TLS Access Clean-up",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Domain Reassignment Pass",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Optimize machine instruction PHIs",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Early If-Conversion",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 cmov Conversion",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Common Subexpression Elimination",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine code sinking",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Peephole Optimizations",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Live Range Shrink",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Optimize",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Optimize Call Frame",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Two-Address instruction pass",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Instruction Scheduler",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Shrink Wrapping analysis",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Control Flow Optimizer",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Copy Propagation Pass",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Post RA top-down list latency scheduler",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Branch Probability Basic Block Placement",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Execution Dependency Fix",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Byte/Word Instruction Fixup",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Atom pad short functions",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Fixup",
      target="_Z16show_short_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 DAG->DAG Instruction Selection",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Local Dynamic TLS Access Clean-up",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Domain Reassignment Pass",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Optimize machine instruction PHIs",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Early If-Conversion",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 cmov Conversion",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Common Subexpression Elimination",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine code sinking",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Peephole Optimizations",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Live Range Shrink",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Optimize",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Optimize Call Frame",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Two-Address instruction pass",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Instruction Scheduler",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Shrink Wrapping analysis",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Control Flow Optimizer",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Copy Propagation Pass",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Post RA top-down list latency scheduler",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Branch Probability Basic Block Placement",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Execution Dependency Fix",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Byte/Word Instruction Fixup",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Atom pad short functions",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Fixup",
      target="_Z15show_full_boardP7NQueensPi",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 DAG->DAG Instruction Selection",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Local Dynamic TLS Access Clean-up",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Domain Reassignment Pass",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Optimize machine instruction PHIs",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Early If-Conversion",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 cmov Conversion",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Common Subexpression Elimination",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine code sinking",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Peephole Optimizations",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Live Range Shrink",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Optimize",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Optimize Call Frame",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Two-Address instruction pass",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Instruction Scheduler",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Shrink Wrapping analysis",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Control Flow Optimizer",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Copy Propagation Pass",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Post RA top-down list latency scheduler",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Branch Probability Basic Block Placement",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Execution Dependency Fix",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Byte/Word Instruction Fixup",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Atom pad short functions",
      target="_Z11check_placePiii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Fixup", target="_Z11check_placePiii", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 DAG->DAG Instruction Selection",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Local Dynamic TLS Access Clean-up",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Domain Reassignment Pass",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Optimize machine instruction PHIs",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Early If-Conversion",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 cmov Conversion",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Common Subexpression Elimination",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine code sinking",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Peephole Optimizations",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Live Range Shrink",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Optimize",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Optimize Call Frame",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Two-Address instruction pass",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Instruction Scheduler",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Shrink Wrapping analysis",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Control Flow Optimizer",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Copy Propagation Pass",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Post RA top-down list latency scheduler",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Branch Probability Basic Block Placement",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Execution Dependency Fix",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Byte/Word Instruction Fixup",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Atom pad short functions",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Fixup",
      target="_Z9put_queenP7NQueensPii",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 DAG->DAG Instruction Selection",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Local Dynamic TLS Access Clean-up",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Domain Reassignment Pass",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Optimize machine instruction PHIs",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Early If-Conversion",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 cmov Conversion",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Common Subexpression Elimination",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine code sinking",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Peephole Optimizations",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Live Range Shrink",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Optimize",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Optimize Call Frame",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Two-Address instruction pass",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Instruction Scheduler",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Shrink Wrapping analysis",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Control Flow Optimizer",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Copy Propagation Pass",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Post RA top-down list latency scheduler",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Branch Probability Basic Block Placement",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Execution Dependency Fix",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Byte/Word Instruction Fixup",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Atom pad short functions",
      target="_Z5solveP7NQueens",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Fixup", target="_Z5solveP7NQueens", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 DAG->DAG Instruction Selection",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Local Dynamic TLS Access Clean-up",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Domain Reassignment Pass", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Optimize machine instruction PHIs",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Early If-Conversion", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 cmov Conversion", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Common Subexpression Elimination",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine code sinking", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Peephole Optimizations", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Remove dead machine instructions",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Live Range Shrink", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Optimize", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 Optimize Call Frame", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Two-Address instruction pass", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Machine Instruction Scheduler",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Machine Loop Invariant Code Motion",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Shrink Wrapping analysis", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Control Flow Optimizer", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Tail Duplication", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="Machine Copy Propagation Pass",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Post RA top-down list latency scheduler",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="Branch Probability Basic Block Placement",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Execution Dependency Fix", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 Byte/Word Instruction Fixup",
      target="main",
      target_type="function",
    ),
    clang.OptPassRunInvocation(
      name="X86 Atom pad short functions", target="main", target_type="function"
    ),
    clang.OptPassRunInvocation(
      name="X86 LEA Fixup", target="main", target_type="function"
    ),
  ]


if __name__ == "__main__":
  test.Main()
