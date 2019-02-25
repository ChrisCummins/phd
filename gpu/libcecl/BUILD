# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of libcecl.
#
# libcecl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libcecl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libcecl.  If not, see <https://www.gnu.org/licenses/>.

licenses(["restricted"])  # GPL v3

exports_files([
    "LICENSE",
    "README.md",
])

filegroup(
    name = "libcecl_header",
    srcs = ["libcecl.h"],
    visibility = ["//visibility:public"],
)

# Two targets for the same code, but for different purposes. :libcecl is a
# library that can be used in the deps attribute of cc_binary targets that need
# to link against libcecl. The :libcecl.so target produces a shared object which
# can be used in the data attribute of other targets.

cc_library(
    name = "libcecl",
    srcs = ["libcecl.c"],
    hdrs = ["libcecl.h"],
    copts = [],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libcecl.so",
    srcs = [
        "libcecl.c",
        "libcecl.h",
    ],
    copts = [],
    linkopts = select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": ["-lOpenCL"],
    }),
    linkshared = 1,
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

py_binary(
    name = "libcecl_rewriter",
    srcs = ["libcecl_rewriter.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:fs",
        "//third_party/py/absl",
    ],
)

py_test(
    name = "libcecl_rewriter_test",
    srcs = ["libcecl_rewriter_test.py"],
    deps = [
        ":libcecl_rewriter",
        "//labm8:test",
    ],
)

py_library(
    name = "libcecl_compile",
    srcs = ["libcecl_compile.py"],
    data = [
        "@opencl_120_headers//:opencl_headers",
        "//gpu/libcecl:libcecl.so",
        "//gpu/libcecl:libcecl_header",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL.so"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8:bazelutil",
        "//labm8:system",
        "//third_party/py/absl",
    ],
)

py_library(
    name = "libcecl_runtime",
    srcs = ["libcecl_runtime.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":libcecl_compile",
        "//gpu/cldrive/legacy:env",
        "//gpu/libcecl/proto:libcecl_py_pb2",
        "//labm8:labdate",
        "//third_party/py/absl",
    ],
)

sh_binary(
    name = "mkcecl",
    srcs = ["mkcecl.sh"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "runcecl",
    srcs = ["runcecl.sh"],
    visibility = ["//visibility:public"],
)
