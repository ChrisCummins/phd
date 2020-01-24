# Control Flow Graphs.
#
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

py_library(
    name = "control_flow_graph",
    srcs = ["control_flow_graph.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/py:app",
        "//third_party/py/networkx",
    ],
)

py_test(
    name = "control_flow_graph_test",
    srcs = ["control_flow_graph_test.py"],
    deps = [
        ":control_flow_graph",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/networkx",
    ],
)

py_library(
    name = "llvm_util",
    srcs = ["llvm_util.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":control_flow_graph",
        "//compilers/llvm:opt_util",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:pbutil",
        "//third_party/py/networkx",
        "//third_party/py/pydot",
        "//third_party/py/pyparsing",
    ],
)

py_test(
    name = "llvm_util_test",
    srcs = ["llvm_util_test.py"],
    deps = [
        ":llvm_util",
        "//compilers/llvm:opt",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)
