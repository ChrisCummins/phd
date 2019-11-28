# Does CLgen output programs with Grewe et al features close to benchmarks?

py_library(
    name = "grewe_features_db",
    srcs = ["grewe_features_db.py"],
    visibility = ["//experimental/deeplearning/clgen:__subpackages__"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:humanize",
        "//labm8/py:sqlutil",
        "//research/grewe_2013_cgo:feature_extractor",
        "//third_party/py/pandas",
        "//third_party/py/progressbar",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "make_cpu_gpu_set",
    srcs = ["make_cpu_gpu_set.py"],
    deps = [
        "//experimental/deeplearning/clgen/closeness_to_grewe_features:grewe_features_db",
        "//labm8/py:app",
        "//labm8/py:prof",
        "//labm8/py:sqlutil",
        "//third_party/py/sqlalchemy",
    ],
)
