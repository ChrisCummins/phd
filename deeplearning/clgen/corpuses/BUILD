# This package implements corpuses for training CLgen.
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

py_library(
    name = "atomizers",
    srcs = ["atomizers.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen:errors",
        "//labm8/py:app",
        "//labm8/py:labdate",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "atomizers_test",
    srcs = ["atomizers_test.py"],
    deps = [
        ":atomizers",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_library(
    name = "corpuses",
    srcs = ["corpuses.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":atomizers",
        ":encoded",
        ":preprocessed",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/dashboard:dashboard_db",
        "//deeplearning/clgen/preprocessors",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:crypto",
        "//labm8/py:hashcache",
        "//labm8/py:humanize",
        "//labm8/py:lockfile",
        "//labm8/py:pbutil",
        "//labm8/py:prof",
        "//third_party/py/checksumdir",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "corpuses_test",
    srcs = ["corpuses_test.py"],
    deps = [
        ":corpuses",
        ":encoded",
        ":preprocessed",
        "//deeplearning/clgen:conftest",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors:common",
        "//deeplearning/clgen/preprocessors:cxx",
        "//deeplearning/clgen/preprocessors:opencl",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)

py_library(
    name = "encoded",
    srcs = ["encoded.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":atomizers",
        ":preprocessed",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:sqlutil",
        "//third_party/py/numpy",
        "//third_party/py/progressbar",
        "//third_party/py/sqlalchemy",
    ],
)

py_test(
    name = "encoded_test",
    srcs = ["encoded_test.py"],
    deps = [
        ":atomizers",
        ":encoded",
        ":preprocessed",
        "//deeplearning/clgen:errors",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_library(
    name = "preprocessed",
    srcs = ["preprocessed.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/preprocessors",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:fs",
        "//labm8/py:sqlutil",
        "//third_party/py/progressbar",
        "//third_party/py/sqlalchemy",
    ],
)
