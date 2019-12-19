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
"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from labm8.py import app


FLAGS = app.FLAGS


def test_FindCallSites_multiple_call_sites():
  builder = programl.GraphBuilder()
  fn1 = builder.AddFunction()
  call = builder.AddNode(function=fn1, text="%2 = call i32 @B()")
  foo = builder.AddNode(function=fn1)
  call2 = builder.AddNode(function=fn1, text="%call = call i32 @B()")
  g = builder.g

  call_sites = llvm_statements.FindCallSites(g, 1, "B")
  assert len(call_sites) == 2
  assert set(call_sites) == {"call", "call2"}
