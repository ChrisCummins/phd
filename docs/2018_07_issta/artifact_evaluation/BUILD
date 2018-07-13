# Artifact for the ISSTA'18 paper "Compiler Fuzzing through Deep Learning".

load("@requirements//:requirements.bzl", "requirement")
load("@io_bazel_rules_docker//container:container.bzl", "container_push")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

licenses(["restricted"])  # GPL v3

exports_files(["LICENSE"])

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
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//lib/labm8:bazelutil",
        "//lib/labm8:crypto",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
    ],
)

py_test(
    name = "01_evaluate_generator_test",
    srcs = ["01_evaluate_generator_test.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":01_evaluate_generator",
        "//deeplearning/clgen:conftest",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:generator_py_pb2",
        "//lib/labm8:pbutil",
        requirement("absl-py"),
        requirement("pytest"),
    ],
)

py_binary(
    name = "02_evaluate_harness",
    srcs = ["02_evaluate_harness.py"],
    data = glob(["data/testcases/*.pbtxt"]),
    default_python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//deeplearning/deepsmith/harnesses:cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_py_pb2",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//gpu/oclgrind",
        "//lib/labm8:bazelutil",
        "//lib/labm8:crypto",
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
        "//third_party/py/pandas",
        requirement("absl-py"),
        requirement("progressbar2"),
    ],
)

# Docker images.

py3_image(
    name = "01_evaluate_generator_image",
    srcs = ["01_evaluate_generator.py"],
    base = "@base//image",
    main = "01_evaluate_generator.py",
    deps = [":01_evaluate_generator"],
)

container_push(
    name = "push_01_evaluate_generator_image",
    format = "Docker",
    image = ":01_evaluate_generator_image",
    registry = "index.docker.io",
    repository = "chriscummins/compiler_fuzzing_through_deep_learning_01",
    tag = "latest",
)

py3_image(
    name = "02_evaluate_harness_image",
    srcs = ["02_evaluate_harness.py"],
    base = "@base//image",
    main = "02_evaluate_harness.py",
    deps = [":02_evaluate_harness"],
)

py3_image(
    name = "03_evaluate_results_image",
    srcs = ["03_evaluate_results.py"],
    base = "@base//image",
    main = "03_evaluate_results.py",
    deps = [":03_evaluate_results"],
)
