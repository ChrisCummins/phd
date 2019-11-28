# A CLgen model which uses backtracking during sampling.

py_binary(
    name = "backtracking",
    srcs = ["backtracking.py"],
    deps = [
        ":backtracking_model",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen/preprocessors",
        "//labm8/py:app",
        "//research/cummins_2017_cgo:generative_model",
    ],
)

sh_test(
    name = "backtracking_smoke_test",
    srcs = ["backtracking_smoke_test.sh"],
    data = [
        ":backtracking",
    ],
)

py_library(
    name = "backtracking_db",
    srcs = ["backtracking_db.py"],
    visibility = ["//experimental/deeplearning/clgen:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:sqlutil",
        "//research/grewe_2013_cgo:feature_extractor",
        "//third_party/py/numpy",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "backtracking_model",
    srcs = ["backtracking_model.py"],
    deps = [
        ":logger",
        "//compilers/llvm:clang",
        "//deeplearning/clgen:sample_observers",
        "//deeplearning/clgen:samplers",
        "//deeplearning/clgen/corpuses:atomizers",
        "//deeplearning/clgen/models",
        "//deeplearning/clgen/models:tensorflow_backend",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:labdate",
        "//research/grewe_2013_cgo:feature_extractor",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)

py_test(
    name = "backtracking_model_test",
    srcs = ["backtracking_model_test.py"],
    deps = [
        ":backtracking_model",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "logger",
    srcs = ["logger.py"],
    deps = [
        ":backtracking_db",
        "//labm8/py:app",
        "//labm8/py:humanize",
    ],
)

py_binary(
    name = "server",
    srcs = ["server.py"],
    data = [
        "//experimental/deeplearning/clgen/backtracking/static",
        "//experimental/deeplearning/clgen/backtracking/templates",
    ],
    deps = [
        ":backtracking",
        ":backtracking_model",
        ":logger",
        "//:build_info",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:humanize",
        "//labm8/py:prof",
        "//third_party/py/flask",
        "//third_party/py/portpicker",
        "//third_party/py/sqlalchemy",
    ],
)
