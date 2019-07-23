# GPGPU benchmark suites.
#
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

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")
load("@build_stack_rules_proto//python:python_proto_library.bzl", "python_proto_library")

filegroup(
    name = "amd-app-sdk-3.0_files",
    srcs = glob(["amd-app-sdk-3.0/**"]),
)

filegroup(
    name = "npb-3.3_files",
    srcs = glob(["npb-3.3/**"]),
)

filegroup(
    name = "nvidia-4.2_files",
    srcs = glob(["nvidia-4.2/**"]),
)

filegroup(
    name = "parboil-0.2_files",
    srcs = glob(["parboil-0.2/**"]),
)

filegroup(
    name = "polybench-gpu-1.0_files",
    srcs = glob(["polybench-gpu-1.0/**"]),
)

filegroup(
    name = "rodinia-3.1_files",
    srcs = glob(["rodinia-3.1/**"]) + [
        "@rodinia_data//:data_files",
    ],
)

filegroup(
    name = "shoc-1.1.5_files",
    srcs = glob(["shoc-1.1.5/**"]),
)

proto_library(
    name = "gpgpu_pb",
    srcs = ["gpgpu.proto"],
    deps = [
        "//gpu/libcecl/proto:libcecl_pb",
    ],
)

python_proto_library(
    name = "gpgpu_pb_py",
    visibility = ["//visibility:public"],
    deps = [":gpgpu_pb"],
)

py_binary(
    name = "gpgpu",
    srcs = ["gpgpu.py"],
    data = [
        ":amd-app-sdk-3.0_files",
        ":npb-3.3_files",
        ":nvidia-4.2_files",
        ":parboil-0.2_files",
        ":polybench-gpu-1.0_files",
        ":rodinia-3.1_files",
        ":shoc-1.1.5_files",
        "//datasets/benchmarks/gpgpu/dummy_just_for_testing:hello_sources",
        "//gpu/libcecl:mkcecl",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":gpgpu_pb_py",
        "//gpu/cldrive/legacy:env",
        "//gpu/libcecl:libcecl_runtime",
        "//gpu/oclgrind",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//labm8:system",
        "//labm8:text",
    ],
)

py_binary(
    name = "gpgpu_verify",
    srcs = ["gpgpu_verify.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpgpu_pb_py",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:pbutil",
    ],
)

py_test(
    name = "gpgpu_test",
    srcs = ["gpgpu_test.py"],
    deps = [
        ":gpgpu",
        ":gpgpu_pb_py",
        "//labm8:app",
        "//labm8:test",
    ],
)

bats_test(
    name = "gpgpu_smoke_test",
    srcs = ["gpgpu_smoke_test.bats"],
    data = [
        ":gpgpu",
        "//labm8/sh:test",
    ],
)
