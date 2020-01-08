# This package implements the machine learning models for CLgen.
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
    name = "backends",
    srcs = ["backends.py"],
    deps = [
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen/dashboard",
        "//deeplearning/clgen/dashboard:dashboard_db",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:cache",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "builders",
    srcs = ["builders.py"],
    deps = [
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:pbutil",
        "//third_party/py/keras",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "builders_test",
    srcs = ["builders_test.py"],
    deps = [
        ":builders",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_library(
    name = "data_generators",
    srcs = ["data_generators.py"],
    deps = [
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "data_generators_test",
    srcs = ["data_generators_test.py"],
    deps = [
        ":data_generators",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_library(
    name = "keras_backend",
    srcs = ["keras_backend.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":backends",
        ":builders",
        ":data_generators",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:crypto",
        "//labm8/py:humanize",
        "//labm8/py:labdate",
        "//labm8/py:logutil",
        "//labm8/py:pbutil",
        "//third_party/py/keras",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "keras_backend_test",
    size = "enormous",
    srcs = ["keras_backend_test.py"],
    deps = [
        ":keras_backend",
        ":models",
        "//deeplearning/clgen:sample_observers",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":builders",
        ":keras_backend",
        ":tensorflow_backend",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen:sample_observers",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses",
        "//deeplearning/clgen/dashboard:dashboard_db",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:humanize",
        "//labm8/py:lockfile",
        "//labm8/py:pbutil",
        "//labm8/py:system",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "pretrained",
    srcs = ["pretrained.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":builders",
        ":keras_backend",
        ":tensorflow_backend",
        "//deeplearning/clgen:cache",
        "//deeplearning/clgen:errors",
        "//deeplearning/clgen:sample_observers",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:cache",
        "//labm8/py:humanize",
        "//labm8/py:labdate",
        "//labm8/py:pbutil",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    deps = [
        ":models",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)

py_library(
    name = "tensorflow_backend",
    srcs = [
        "helper.py",
        "tensorflow_backend.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":backends",
        ":builders",
        ":data_generators",
        "//deeplearning/clgen:sample_observers",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen:telemetry",
        "//deeplearning/clgen/dashboard:dashboard_db",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:humanize",
        "//labm8/py:labdate",
        "//labm8/py:logutil",
        "//labm8/py:pbutil",
        "//third_party/py/numpy",
        "//third_party/py/progressbar",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "tensorflow_backend_test",
    timeout = "long",
    srcs = ["tensorflow_backend_test.py"],
    deps = [
        ":models",
        ":tensorflow_backend",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//deeplearning/clgen/tests:fixtures",
        "//labm8/py:app",
        "//labm8/py:crypto",
        "//labm8/py:fs",
        "//labm8/py:test",
    ],
)
