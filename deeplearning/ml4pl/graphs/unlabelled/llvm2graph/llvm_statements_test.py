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
