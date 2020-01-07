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

filegroup(
    name = "hello_sources",
    srcs = ["hello.cc"],
)

cc_test(
    name = "hello",
    srcs = [":hello_sources"],
    linkopts = [
        "-ldl",
        "-lpthread",
    ] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    linkstatic = False,  # Needed for oclgrind support.
    deps = [
        "//third_party/opencl",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

py_test(
    name = "integration_test",
    srcs = ["integration_test.py"],
    data = [
        ":hello_sources",
        "//gpu/libcecl:libcecl.so",
        "//third_party/clinfo:clinfo_c_src",
    ],
    deps = [
        "//compilers/llvm:clang",
        "//gpu/libcecl:libcecl_rewriter",
        "//gpu/libcecl:libcecl_runtime",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)
