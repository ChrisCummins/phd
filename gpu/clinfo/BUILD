# A package for listing available OpenCL devices.
#
# Copyright 2017, 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
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

cc_binary(
    name = "clinfo",
    srcs = ["clinfo.cc"],
    linkstatic = False,  # Needed for oclgrind support.
    visibility = ["//visibility:public"],
    deps = [":libclinfo"],
)

cc_library(
    name = "libclinfo",
    srcs = [
        "libclinfo.cc",
        "libclinfo.h",
    ],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/clinfo/proto:clinfo_cc_pb2",
        "//third_party/opencl",
        "//phd:string",
        "//phd:logging",
        "//phd:statusor",
        "@com_google_absl//absl/strings",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_test(
    name = "libclinfo_test",
    srcs = ["libclinfo_test.cc"],
    linkstatic = False,  # Needed for oclgrind support.
    deps = [
        ":libclinfo",
        "//phd:test",
        "//third_party/opencl",
    ],
)
