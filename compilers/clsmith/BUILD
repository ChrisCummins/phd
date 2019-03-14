# A python wrapper around CLSmith, a random generator of OpenCL C programs.
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

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "clsmith",
    srcs = ["clsmith.py"],
    data = ["@CLSmith"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
    ],
)

py_test(
    name = "clsmith_test",
    srcs = ["clsmith_test.py"],
    deps = [
        ":clsmith",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_binary(
    name = "cl_launcher",
    srcs = ["cl_launcher.py"],
    data = [
        "@CLSmith//:cl_launcher",
        "@CLSmith//:cl_safe_math_macros",
        "@CLSmith//:runtime_headers",
        "@CLSmith//:safe_math_macros",
    ] + select({
        "//:darwin": [],
        "//conditions:default": [
            "@libopencl//:libOpenCL.so",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/cldrive/legacy:driver",
        "//gpu/cldrive/legacy:env",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:fs",
        "//labm8:system",
    ],
)

py_test(
    name = "cl_launcher_test",
    srcs = ["cl_launcher_test.py"],
    deps = [
        ":cl_launcher",
        "//gpu/cldrive/legacy:env",
        "//labm8:app",
        "//labm8:test",
    ],
)
