# This package contains CLgen, a deep learning program generator.
#
# CLgen is an open source application for generating runnable programs using
# deep learning. CLgen learns to program using neural networks which model the
# semantics and usage from large volumes of program fragments, generating
# executable programs that are representative of, but distinct from,
# the programs it learns from.
#
# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

licenses(["restricted"])  # GPL v3

exports_files([
    "LICENSE",
    "README.md",
])

py_library(
    name = "cache",
    srcs = ["cache.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
    deps = [
        "//labm8:app",
        "//labm8:cache",
        "//labm8:fs",
    ],
)

py_binary(
    name = "clgen",
    srcs = ["clgen.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":errors",
        ":samplers",
        "//deeplearning/clgen/models",
        "//deeplearning/clgen/models:pretrained",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
        "//labm8:prof",
    ],
)

py3_image(
    name = "clgen_image",
    srcs = ["clgen.py"],
    base = "@base//image",
    main = "clgen.py",
    deps = [
        # This list must be kept in sync with :clgen deps.
        ":errors",
        ":samplers",
        "//deeplearning/clgen/models",
        "//deeplearning/clgen/models:pretrained",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
        "//labm8:prof",
    ],
)

py_test(
    name = "clgen_test",
    size = "enormous",
    srcs = ["clgen_test.py"],
    data = [
        "//deeplearning/clgen/tests/data:pico",
        "//deeplearning/clgen/tests/data:tiny",
        "//deeplearning/clgen/tests/data/tiny",
    ],
    deps = [
        ":clgen",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:pbutil",
        "//labm8:tar",
        "//labm8:test",
    ],
)

py_library(
    name = "conftest",
    testonly = True,
    srcs = ["conftest.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//deeplearning/clgen/proto:corpus_py_pb2",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
        "//labm8:test",
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

py_binary(
    name = "ls_models",
    srcs = ["ls_models.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen/models",
        "//deeplearning/clgen/proto:internal_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
    ],
)

py_library(
    name = "sample",
    srcs = ["sample.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":errors",
        ":samplers",
        "//deeplearning/clgen/models:pretrained",
        "//deeplearning/clgen/proto:clgen_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
    ],
)

py_library(
    name = "samplers",
    srcs = ["samplers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cache",
        ":errors",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen/proto:sampler_py_pb2",
        "//labm8:app",
        "//labm8:crypto",
        "//labm8:pbutil",
    ],
)

py_test(
    name = "samplers_test",
    srcs = ["samplers_test.py"],
    deps = [
        ":conftest",
        ":errors",
        ":samplers",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "telemetry",
    srcs = ["telemetry.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen/proto:telemetry_py_pb2",
        "//labm8:app",
        "//labm8:jsonutil",
        "//labm8:labdate",
        "//labm8:pbutil",
    ],
)

py_test(
    name = "telemetry_test",
    srcs = ["telemetry_test.py"],
    deps = [
        ":telemetry",
        "//labm8:app",
        "//labm8:test",
    ],
)
