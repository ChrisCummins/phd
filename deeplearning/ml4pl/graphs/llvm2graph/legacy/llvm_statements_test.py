# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for //deeplearning/ml4pl/graphs/llvm2graph/legacy:llvm_statements."""
from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs.llvm2graph.legacy import llvm_statements
from labm8.py import test

FLAGS = test.FLAGS


def test_FindCallSites_multiple_call_sites():
  builder = programl.GraphBuilder()
  fn1 = builder.AddFunction()
  call = builder.AddNode(function=fn1, text="%2 = call i32 @B()")
  foo = builder.AddNode(function=fn1)
  call2 = builder.AddNode(function=fn1, text="%call = call i32 @B()")
  g = builder.g

  call_sites = llvm_statements.FindCallSites(g, fn1, "B")
  assert len(call_sites) == 2
  assert set(call_sites) == {call, call2}


def test_GetLlvmStatementDefAndUses():
  statement = "%1 = alloca i32, align 4"
  def_, uses = llvm_statements.GetLlvmStatementDefAndUses(statement)
  assert def_ == "%1"
  assert not uses

  statement = "store i32 0, i32* %1, align 4"
  def_, uses = llvm_statements.GetLlvmStatementDefAndUses(statement)
  assert def_ == ""
  assert uses == ["0", "%1"]

  statement = "br label %3"
  def_, uses = llvm_statements.GetLlvmStatementDefAndUses(statement)
  assert def_ == ""
  assert uses == ["%3"]

  statement = "store i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i8** %3, align 8"
  def_, uses = llvm_statements.GetLlvmStatementDefAndUses(statement)
  assert def_ == ""
  assert uses == ["@.str", "0", "0", "%3"]

  statement = "%5 = load i32, i32* %2, align 4"
  def_, uses = llvm_statements.GetLlvmStatementDefAndUses(statement)
  assert def_ == "%5"
  assert uses == ["%2"]

  statement = "%6 = icmp sgt i32 %5, 0"
  def_, uses = llvm_statements.GetLlvmStatementDefAndUses(statement)
  assert def_ == "%6"
  assert uses == ["%5", "0"]

  statement = "br i1 %6, label %7, label %8"
  def_, uses = llvm_statements.GetLlvmStatementDefAndUses(statement)
  assert def_ == ""
  assert uses == ["%6", "%7", "%8"]

  statement = "store float 0x40C80C0F60000000, float* %4, align 4"
  def_, uses = llvm_statements.GetLlvmStatementDefAndUses(statement)
  assert def_ == ""
  assert uses == ["0x40C80C0F60000000", "%4"]

  statement = "%3 = alloca i8*, align 8"
  def_, uses = llvm_statements.GetLlvmStatementDefAndUses(statement)
  assert def_ == "%3"
  assert uses == []


if __name__ == "__main__":
  test.Main()
