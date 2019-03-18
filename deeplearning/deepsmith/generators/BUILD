# Generator implementations.
#
# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.

load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

package(default_visibility = ["//experimental:__subpackages__"])

py_binary(
    name = "clgen",
    srcs = ["clgen.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":clgen_pretrained",
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:model_py_pb2",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//labm8:app",
    ],
)

py_binary(
    name = "clgen_pretrained",
    srcs = ["clgen_pretrained.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":generator",
        "//deeplearning/clgen:sample",
        "//deeplearning/clgen:sample_observers",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//labm8:app",
        "//third_party/py/grpcio",
    ],
)

# TODO(cec): Re-enable once CLgen cyclic symlinks is fixed.
# py3_image(
#     name = "clgen_image",
#     base = "@base//image",
#     srcs = ["clgen.py"],
#     deps = [":clgen"],
#     main = "clgen.py",
# )

py_binary(
    name = "clsmith",
    srcs = ["clsmith.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":generator",
        "//compilers/clsmith",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//deeplearning/deepsmith/proto:service_py_pb2",
        "//labm8:app",
        "//labm8:labdate",
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
    name = "dummy_generator",
    srcs = ["dummy_generator.py"],
    visibility = [
        "//deeplearning/deepsmith/cli:__subpackages__",
        "//experimental:__subpackages__",
    ],
    deps = [
        ":generator",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//labm8:app",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//third_party/py/grpcio",
    ],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//labm8:app",
        "//labm8:pbutil",
        "//third_party/py/grpcio",
    ],
)

py_binary(
    name = "randchar",
    srcs = ["randchar.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":generator",
        "//deeplearning/deepsmith:services",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//labm8:app",
        "//labm8:labdate",
        "//labm8:pbutil",
        "//third_party/py/grpcio",
    ],
)
