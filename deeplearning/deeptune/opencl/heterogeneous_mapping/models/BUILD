# Models for OpenCL hetereogeneous device mapping.
#
# Copyright (c) 2017-2020 Chris Cummins.
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

py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        "//deeplearning/clgen/corpuses:atomizers",
        "//labm8/py:app",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_library(
    name = "deeptune",
    srcs = ["deeptune.py"],
    deps = [
        ":base",
        "//deeplearning/clgen/corpuses:atomizers",
        "//labm8/py:app",
        "//third_party/py/keras",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "deeptune_test",
    timeout = "long",
    srcs = ["deeptune_test.py"],
    deps = [
        ":deeptune",
        ":testlib",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:conftest",
        "//labm8/py:test",
    ],
)

py_library(
    name = "grewe",
    srcs = ["grewe.py"],
    deps = [
        ":base",
        "//datasets/opencl/device_mapping:opencl_device_mapping_dataset",
        "//deeplearning/clgen/corpuses:atomizers",
        "//labm8/py:app",
        "//third_party/py/pandas",
        "//third_party/py/scikit_learn",
    ],
)

py_test(
    name = "grewe_test",
    srcs = ["grewe_test.py"],
    deps = [
        ":grewe",
        ":testlib",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:conftest",
        "//labm8/py:test",
    ],
)

py_library(
    name = "lda",
    srcs = ["lda.py"],
    deps = [
        ":base",
        ":ncc",
        "//compilers/llvm:opt_util",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:utils",
        "//deeplearning/ml4pl/graphs/llvm2graph/legacy/cfg:llvm_util",
        "//deeplearning/ncc:task_utils",
        "//deeplearning/ncc:vocabulary",
        "//labm8/py:app",
        "//third_party/py/graph_nets",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "lda_test",
    timeout = "long",
    srcs = ["lda_test.py"],
    deps = [
        ":models",
        ":testlib",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:conftest",
        "//deeplearning/ml4pl/graphs/llvm2graph/legacy/cfg:llvm_util",
        "//labm8/py:test",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":deeptune",
        ":grewe",
        ":lda",
        ":ncc",
        ":static_mapping",
        "//labm8/py:app",
        "//labm8/py:labtypes",
    ],
)

py_test(
    name = "models_test",
    timeout = "long",
    srcs = ["models_test.py"],
    deps = [
        ":models",
        "//labm8/py:test",
    ],
)

py_library(
    name = "ncc",
    srcs = ["ncc.py"],
    data = [
        "//deeplearning/ncc/published_results:embeddings",
        "//deeplearning/ncc/published_results:task_devmap_kernels",
        "//deeplearning/ncc/published_results:vocabulary",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":deeptune",
        "//compilers/llvm:clang",
        "//deeplearning/clgen/preprocessors:opencl",
        "//deeplearning/ncc:task_utils",
        "//deeplearning/ncc:vocabulary",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "ncc_test",
    timeout = "long",
    srcs = ["ncc_test.py"],
    deps = [
        ":ncc",
        ":testlib",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:conftest",
        "//labm8/py:test",
    ],
)

py_library(
    name = "static_mapping",
    srcs = ["static_mapping.py"],
    deps = [
        ":base",
        "//deeplearning/clgen/corpuses:atomizers",
        "//labm8/py:app",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "static_mapping_test",
    srcs = ["static_mapping_test.py"],
    deps = [
        ":static_mapping",
        ":testlib",
        "//deeplearning/deeptune/opencl/heterogeneous_mapping:conftest",
        "//labm8/py:test",
    ],
)

py_library(
    name = "testlib",
    testonly = 1,
    srcs = ["testlib.py"],
    deps = [
        "//labm8/py:app",
    ],
)
