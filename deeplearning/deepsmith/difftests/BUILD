# Differential tests for DeepSmith.
#
# Copyright (c) 2017-2020 Chris Cummins.
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

py_library(
    name = "difftests",
    srcs = ["difftests.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//labm8/py:app",
    ],
)

py_test(
    name = "difftests_test",
    srcs = ["difftests_test.py"],
    deps = [
        ":difftests",
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

java_binary(
    name = "JavaDifftester",
    srcs = ["JavaDifftester.java"],
    main_class = "deeplearning.deepsmith.difftests.JavaDifftester",
    runtime_deps = [
        # Copied from the runtime_deps of
        # //deeplearning/clgen/preprocessors:JavaPreprocessor.
        # I don't know how to make these runtime deps propagate.
        "@org_apache_commons_cli//jar",
        "@org_apache_commons_io//jar",
        "@org_apache_commons_jci_core//jar",
        "@org_apache_commons_jci_eclipse//jar",
        "@org_apache_commons_jci_examples//jar",
        "@org_apache_commons_logging_api//jar",
        "@org_eclipse_core_contenttype//jar",
        "@org_eclipse_core_runtime//jar",
        "@org_eclipse_equinox_common//jar",
        "@org_eclipse_equinox_preferences//jar",
        "@org_eclipse_jdt_core//jar",
        "@org_eclipse_jface//jar",
        "@org_eclipse_jobs//jar",
        "@org_eclipse_osgi//jar",
        "@org_eclipse_osgi_util//jar",
        "@org_eclipse_resources//jar",
        "@org_eclipse_text//jar",
        "@org_osgi_framework//jar",
        "@org_osgi_service_prefs//jar",
    ],
    deps = [
        "//deeplearning/clgen/preprocessors:JavaPreprocessor",
        "//deeplearning/clgen/proto:clgen_pb_java",
        "//deeplearning/deepsmith/harnesses:JavaDriver",
        "@com_google_guava//jar",
    ],
)

py_library(
    name = "opencl",
    srcs = ["opencl.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":difftests",
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//gpu/cldrive/legacy:args",
        "//labm8/py:app",
    ],
)
