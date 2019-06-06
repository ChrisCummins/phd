# Adverserial counter-examples for sequential modeling of source code.
#
# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepTune is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepTune is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepTune.  If not, see <https://www.gnu.org/licenses/>.

py_binary(
    name = "adversary",
    srcs = ["adversary.py"],
    deps = [
        "//deeplearning/deeptune/opencl/heterogeneous_mapping",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping/models",
        "//labm8:app",
        "//labm8:prof",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_library(
    name = "opencl_deadcode_inserter",
    srcs = ["opencl_deadcode_inserter.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen/preprocessors:opencl",
        "//experimental/compilers/reachability:control_flow_graph_generator",
        "//gpu/cldrive/legacy:args",
        "//labm8:app",
        "//labm8:fmt",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "opencl_deadcode_inserter_test",
    srcs = ["opencl_deadcode_inserter_test.py"],
    deps = [
        ":opencl_deadcode_inserter",
        "//compilers/llvm:clang",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "opencl_deadcode_inserter_fuzz_test",
    size = "enormous",
    srcs = ["opencl_deadcode_inserter_fuzz_test.py"],
    deps = [
        ":opencl_deadcode_inserter",
        "//deeplearning/deepsmith/harnesses:cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//gpu/oclgrind",
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/numpy",
    ],
)
