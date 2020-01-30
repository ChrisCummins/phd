# Copyright (c) 2016-2020 Chris Cummins.
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

py_test(
    name = "integration_test",
    srcs = ["integration_test.py"],
    data = [
        "//gpu/libcecl:libcecl.so",
        "//third_party/clinfo:clinfo_c_src",
        "//third_party/opencl/examples:hello_sources",
    ],
    deps = [
        "//compilers/llvm:clang",
        "//gpu/libcecl:libcecl_rewriter",
        "//gpu/libcecl:libcecl_runtime",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)
