# This package contains the data for a tiny CLgen instance.
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

filegroup(
    name = "tiny",
    srcs = [
        "config.pbtxt",
        "corpus.tar.bz2",
    ],
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    data = [":tiny"],
    deps = [
        "//deeplearning/clgen",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//labm8:test",
    ],
)
