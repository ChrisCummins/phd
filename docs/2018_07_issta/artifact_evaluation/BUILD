# This package contains the artifact for my ISSTA'18 paper.
# !!! WARNING THIS PACKAGE CANNOT BE MOVED. !!!
# The scripts have been hardcoded to reference this location within the phd
# repo, i.e. <root>/docs/2018_07_issta/artifact_evaluation.
#
# None of these targets are safe to run using bazel sandboxing. They are merely
# recorded here to track dependencies.

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    data = [
        "//:config_script",
        "//tools:protoc",
    ],
)

py_binary(
    name = "01_evaluate_generator_make_testcases",
    srcs = ["01_evaluate_generator/make_testcases.py"],
    default_python_version = "PY3",
    main = "01_evaluate_generator/make_testcases.py",
    srcs_version = "PY3",
)

sh_binary(
    name = "01_evaluate_generator",
    srcs = ["01_evaluate_generator/run.sh"],
    data = [":01_evaluate_generator_make_testcases"],
)

py_binary(
    name = "02_evaluate_harness_experiments",
    srcs = ["02_evaluate_harness/experiments.py"],
    data = glob(["02_evaluate_harness/data/testcases/*.pbtxt"]),
    default_python_version = "PY3",
    main = "02_evaluate_harness/experiments.py",
    srcs_version = "PY3",
)

sh_binary(
    name = "02_evaluate_harness",
    srcs = ["02_evaluate_harness/run.sh"],
    data = [":02_evaluate_harness_experiments"],
)

py_binary(
    name = "03_evaluate_results_experiments",
    srcs = ["03_evaluate_results/experiments.py"],
    data = glob(["03_evaluate_results/data/**/*.pbtxt"]),
    default_python_version = "PY3",
    main = "03_evaluate_results/experiments.py",
    srcs_version = "PY3",
)

sh_binary(
    name = "03_evaluate_results",
    srcs = ["03_evaluate_results/run.sh"],
    data = [":03_evaluate_results_experiments"],
)
