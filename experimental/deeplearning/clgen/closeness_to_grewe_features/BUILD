# Does CLgen output programs with Grewe et al features close to benchmarks?

py_library(
    name = "grewe_features_db",
    srcs = ["grewe_features_db.py"],
    visibility = ["//experimental/deeplearning/clgen:__subpackages__"],
    deps = [
        "//labm8:sqlutil",
        "//research/grewe_2013_cgo:feature_extractor",
        "//third_party/py/absl",
        "//third_party/py/sqlalchemy",
    ],
)
