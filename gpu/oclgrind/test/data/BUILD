# Oclgrind test data.
#
# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
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

# A small program which checks that it is running inside an Oclgrind
# environment. Use this as a template for other OpenCL targets.
cc_binary(
    name = "oclgrind_working",
    testonly = 1,
    srcs = ["oclgrind_working.cc"],
    linkstatic = False,  # Needed for oclgrind support.
    visibility = ["//gpu/oclgrind:__subpackages__"],
    deps = [
        "//phd:logging",
        "//third_party/opencl",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)
