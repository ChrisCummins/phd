# Artifact for the ISSTA'18 paper "Compiler Fuzzing through Deep Learning".

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
    deps = [
        "//deeplearning/deepsmith/generators:clgen",
        "//deeplearning/deepsmith/proto:generator_pb_py_grpc",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:crypto",
        "//labm8/py:pbutil",
    ],
)

py_test(
    name = "01_evaluate_generator_test",
    srcs = ["01_evaluate_generator_test.py"],
    deps = [
        ":01_evaluate_generator",
        "//deeplearning/clgen:conftest",
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//deeplearning/deepsmith/proto:generator_pb_py_grpc",
        "//labm8/py:app",
        "//labm8/py:pbutil",
        "//labm8/py:test",
    ],
)

py_binary(
    name = "02_evaluate_harness",
    srcs = ["02_evaluate_harness.py"],
    data = glob(["data/testcases/*.pbtxt"]),
    deps = [
        "//deeplearning/deepsmith/harnesses:cldrive",
        "//deeplearning/deepsmith/proto:deepsmith_pb_py_grpc",
        "//deeplearning/deepsmith/proto:harness_py_pb2",
        "//gpu/cldrive/legacy:env",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:crypto",
        "//labm8/py:fs",
        "//labm8/py:labtypes",
        "//labm8/py:pbutil",
    ],
)

py_binary(
    name = "03_evaluate_results",
    srcs = ["03_evaluate_results.py"],
    data = glob([
        "data/datastore.pbtxt",
        "data/our_results/*.pbtxt",
    ]),
    deps = [
        "//deeplearning/deepsmith:datastore",
        "//deeplearning/deepsmith:db",
        "//deeplearning/deepsmith:result",
        "//deeplearning/deepsmith:testbed",
        "//deeplearning/deepsmith:testcase",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:fs",
        "//labm8/py:labtypes",
        "//labm8/py:pbutil",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
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
