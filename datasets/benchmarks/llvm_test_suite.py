# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
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
"""Benchmarks in the LLVM test suite.

See: https://llvm.org/docs/TestingGuide.html#test-suite-overview
"""
import typing

from datasets.benchmarks.proto import benchmarks_pb2
from labm8.py import app
from labm8.py import bazelutil

FLAGS = app.FLAGS


class _SingleSource_Benchmarks_McGill(object):
  """The McGill benchmarks."""
  queens = benchmarks_pb2.Benchmark(
      name='queens',
      usage='queens [-ac] <n>',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_McGill_queens')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/McGill/queens.c')),
      ],
  )


class _SingleSource_Benchmarks_Shootout(object):
  """The Programming Language Shootout benchmarks."""
  ackermann = benchmarks_pb2.Benchmark(
      name='ackermann',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_ackermann')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/ackermann.c'
              )),
      ],
  )
  ary3 = benchmarks_pb2.Benchmark(
      name='ary3',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_ary3')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/ary3.c')),
      ],
  )
  fib2 = benchmarks_pb2.Benchmark(
      name='fib2',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_fib2')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/fib2.c')),
      ],
  )
  hash = benchmarks_pb2.Benchmark(
      name='hash',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_hash')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/hash.c')),
      ],
      hdrs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/simple_hash.h'
              )),
      ],
  )
  heapsort = benchmarks_pb2.Benchmark(
      name='heapsort',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_heapsort')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/heapsort.c')
          ),
      ],
  )
  hello = benchmarks_pb2.Benchmark(
      name='hello',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_hello')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/hello.c')),
      ],
  )
  lists = benchmarks_pb2.Benchmark(
      name='lists',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_lists')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/lists.c')),
      ],
  )
  matrix = benchmarks_pb2.Benchmark(
      name='matrix',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_matrix')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/matrix.c')),
      ],
  )
  methcall = benchmarks_pb2.Benchmark(
      name='methcall',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_methcall')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/methcall.c')
          ),
      ],
  )
  nestedloop = benchmarks_pb2.Benchmark(
      name='nestedloop',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_nestedloop')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/nestedloop.c'
              )),
      ],
  )
  objinst = benchmarks_pb2.Benchmark(
      name='objinst',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_objinst')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/objinst.c')
          ),
      ],
  )
  random = benchmarks_pb2.Benchmark(
      name='random',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_random')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/random.c')),
      ],
  )
  sieve = benchmarks_pb2.Benchmark(
      name='sieve',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_sieve')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/sieve.c')),
      ],
  )
  strcat = benchmarks_pb2.Benchmark(
      name='strcat',
      binary=str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource_Benchmarks_Shootout_strcat')),
      srcs=[
          str(
              bazelutil.DataPath(
                  'llvm_test_suite/SingleSource/Benchmarks/Shootout/strcat.c')),
      ],
  )


class _SingleSource_Benchmarks(object):
  """Single source benchmarks."""
  McGill = _SingleSource_Benchmarks_McGill()
  Shootout = _SingleSource_Benchmarks_Shootout()


class _SingleSource(object):
  """The single source files."""
  Benchmarks = _SingleSource_Benchmarks()


SingleSource = _SingleSource()

# A list of all benchmarks defined in this file.
BENCHMARKS: typing.List[benchmarks_pb2.Benchmark] = [
    SingleSource.Benchmarks.McGill.queens,
    SingleSource.Benchmarks.Shootout.ackermann,
    SingleSource.Benchmarks.Shootout.ary3,
    SingleSource.Benchmarks.Shootout.fib2,
    SingleSource.Benchmarks.Shootout.hash,
    SingleSource.Benchmarks.Shootout.heapsort,
    SingleSource.Benchmarks.Shootout.hello,
    SingleSource.Benchmarks.Shootout.lists,
    SingleSource.Benchmarks.Shootout.matrix,
    SingleSource.Benchmarks.Shootout.methcall,
    SingleSource.Benchmarks.Shootout.nestedloop,
    SingleSource.Benchmarks.Shootout.objinst,
    SingleSource.Benchmarks.Shootout.random,
    SingleSource.Benchmarks.Shootout.sieve,
    SingleSource.Benchmarks.Shootout.strcat,
]
