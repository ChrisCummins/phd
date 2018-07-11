# Artifact for the ISSTA'18 paper "Compiler Fuzzing through Deep Learning".

load("@requirements//:requirements.bzl", "requirement")

py_binary(
    name = "01_evaluate_generator",
    srcs = ["01_evaluate_generator.py"],
    data = [
        "data/clgen.pbtxt",
        "//deeplearning/clgen/tests/data:tiny",
    ],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//deeplearning/deepsmith/generators:clgen",
        "//deeplearning/deepsmith/generators:generator",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//lib/labm8:bazelutil",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
    ],
)

py_binary(
    name = "02_evaluate_harness",
    srcs = ["02_evaluate_harness.py"],
    data = glob(["data/testcases/*.pbtxt"]),
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//deeplearning/deepsmith/generators:clgen",
        "//deeplearning/deepsmith/generators:generator",
        "//deeplearning/deepsmith/harnesses:cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//gpu/oclgrind",
        "//lib/labm8:bazelutil",
        "//lib/labm8:fs",
        "//lib/labm8:labtypes",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
    ],
)

py_binary(
    name = "03_evaluate_results",
    srcs = ["03_evaluate_results.py"],
    data = glob([
        "data/datastore.pbtxt",
        "data/our_results/*.pbtxt",
    ]),
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:db",
        "//deeplearning/deepsmith:result",
        "//deeplearning/deepsmith:testbed",
        "//deeplearning/deepsmith:testcase",
        "//lib/labm8:bazelutil",
        "//lib/labm8:fs",
        "//lib/labm8:labtypes",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
        requirement("pandas"),
        requirement("progressbar2"),
    ],
)
