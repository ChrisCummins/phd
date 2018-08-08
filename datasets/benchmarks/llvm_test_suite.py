"""Benchmarks in the LLVM test suite.

See: https://llvm.org/docs/TestingGuide.html#test-suite-overview
"""
import typing
from absl import flags

from datasets.benchmarks.proto import benchmarks_pb2
from lib.labm8 import bazelutil


FLAGS = flags.FLAGS


class McGillBenchmarks(object):
  """The McGill Benchmarks."""
  queens = benchmarks_pb2.Benchmark(
      name='queens',
      usage='queens [-ac] <n>',
      binary=[bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_McGill_queens')],
      srcs=[str(bazelutil.DataPath(
          'llvm_test_suite/SingleSource/Benchmarks/McGill/queens.c'))],
  )


class ShootoutBenchmarks(object):
  """The Programming Language Shootout benchmarks."""
  ackermann = benchmarks_pb2.Benchmark(
      name='ackermann',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_ackermann'),
      srcs=[str(bazelutil.DataPath(
          'llvm_test_suite/SingleSource/Benchmarks/Shootout/ackermann.c'))]
  )
  ary3 = benchmarks_pb2.Benchmark(
      name='ary3',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_ary3'),
      srcs=[
        str(bazelutil.DataPath(
            'llvm_test_suite/SingleSource/Benchmarks/Shootout/ary3.c'))]
  )
  fib2 = benchmarks_pb2.Benchmark(
      name='fib2',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_fib2'),
      srcs=[
        str(bazelutil.DataPath(
            'llvm_test_suite/SingleSource/Benchmarks/Shootout/fib2.c'))]
  )
  hash = benchmarks_pb2.Benchmark(
      name='hash',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_hash'),
      srcs=[
        str(bazelutil.DataPath(
            'llvm_test_suite/SingleSource/Benchmarks/Shootout/hash.c'))]
  )
  heapsort = benchmarks_pb2.Benchmark(
      name='heapsort',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_heapsort'),
      srcs=[str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource/Benchmarks/Shootout/heapsort.c'))]
  )
  hello = benchmarks_pb2.Benchmark(
      name='hello',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_hello'),
      srcs=[str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource/Benchmarks/Shootout/hello.c'))]
  )
  lists = benchmarks_pb2.Benchmark(
      name='lists',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_lists'),
      srcs=[str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource/Benchmarks/Shootout/lists.c'))]
  )
  matrix = benchmarks_pb2.Benchmark(
      name='matrix',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_matrix'),
      srcs=[str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource/Benchmarks/Shootout/matrix.c'))]
  )
  methcall = benchmarks_pb2.Benchmark(
      name='methcall',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_methcall'),
      srcs=[str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource/Benchmarks/Shootout/methcall.c'))]
  )
  nestedloop = benchmarks_pb2.Benchmark(
      name='nestedloop',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_nestedloop'),
      srcs=[str(bazelutil.DataPath(
          'llvm_test_suite/SingleSource/Benchmarks/Shootout/nestedloop.c'))]
  )
  objinst = benchmarks_pb2.Benchmark(
      name='objinst',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_objinst'),
      srcs=[str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource/Benchmarks/Shootout/objinst.c'))]
  )
  random = benchmarks_pb2.Benchmark(
      name='random',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_random'),
      srcs=[str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource/Benchmarks/Shootout/random.c'))]
  )
  sieve = benchmarks_pb2.Benchmark(
      name='sieve',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_sieve'),
      srcs=[str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource/Benchmarks/Shootout/sieve.c'))]
  )
  strcat = benchmarks_pb2.Benchmark(
      name='strcat',
      binary=bazelutil.DataPath(
          'llvm_test_suite/SingleSource_Benchmarks_Shootout_strcat'),
      srcs=[str(
          bazelutil.DataPath(
              'llvm_test_suite/SingleSource/Benchmarks/Shootout/strcat.c'))]
  )


class SingleSourceBenchmarks(object):
  """The single source benchmarks."""
  McGill = McGillBenchmarks()
  Shootout = ShootoutBenchmarks()


SingleSource = SingleSourceBenchmarks()

# A list of all benchmarks defined in this file.
BENCHMARKS: typing.List[benchmarks_pb2.Benchmark] = [
  SingleSource.McGill.queens,
  SingleSource.Shootout.ackermann,
  SingleSource.Shootout.ary3,
  SingleSource.Shootout.fib2,
  SingleSource.Shootout.hash,
  SingleSource.Shootout.heapsort,
  SingleSource.Shootout.hello,
  SingleSource.Shootout.lists,
  SingleSource.Shootout.matrix,
  SingleSource.Shootout.methcall,
  SingleSource.Shootout.nestedloop,
  SingleSource.Shootout.objinst,
  SingleSource.Shootout.random,
  SingleSource.Shootout.sieve,
  SingleSource.Shootout.strcat,
]
