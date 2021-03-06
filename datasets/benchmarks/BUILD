# Benchmarks.
#
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

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "bzip2",
    srcs = ["bzip2.py"],
    data = [
        "@bzip2",
        "@bzip2//:bzip2_srcs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//datasets/benchmarks/proto:benchmarks_pb_py",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
    ],
)

py_test(
    name = "bzip2_test",
    srcs = ["bzip2_test.py"],
    deps = [
        ":bzip2",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "llvm_test_suite",
    srcs = ["llvm_test_suite.py"],
    data = [
        "@llvm_test_suite//:SingleSource_Benchmarks_McGill_queens",
        "@llvm_test_suite//:SingleSource_Benchmarks_McGill_queens_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_ackermann",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_ackermann_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_ary3",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_ary3_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_fib2",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_fib2_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_hash",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_hash_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_heapsort",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_heapsort_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_hello",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_hello_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_lists",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_lists_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_matrix",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_matrix_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_methcall",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_methcall_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_nestedloop",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_nestedloop_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_objinst",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_objinst_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_random",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_random_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_sieve",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_sieve_srcs",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_strcat",
        "@llvm_test_suite//:SingleSource_Benchmarks_Shootout_strcat_srcs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//datasets/benchmarks/proto:benchmarks_pb_py",
        "//labm8/py:bazelutil",
    ],
)

py_test(
    name = "llvm_test_suite_test",
    srcs = ["llvm_test_suite_test.py"],
    deps = [
        ":llvm_test_suite",
        "//compilers/llvm:clang",
        "//datasets/benchmarks/proto:benchmarks_pb_py",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

filegroup(
    name = "nqueens_src",
    srcs = ["nqueens.cc"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "nqueens",
    srcs = [":nqueens_src"],
)
