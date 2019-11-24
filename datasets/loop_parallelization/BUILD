# A dataset of loop parallelizability.
#
# This is author-provided data from the paper:
#
#     Maramzin, A., Vasiladiotis, C., Lozano, R. C., & Cole, M. (2019).
#     “It Looks Like You’re Writing a Parallel Loop” A Machine Learning
#     Based Parallelization Assistant. In AI-SEPS.
#
# Results hosted at:
#
#     https://github.com/av-maramzin/reports/tree/7f485151f1975000284db108cf7b0938dd6ec374/ppar-metrics/results

filegroup(
    name = "csv",
    srcs = ["loop_parallelization.csv"],
)

py_library(
    name = "loop_parallelization",
    srcs = ["loop_parallelization.py"],
    data = [":csv"],
    deps = [
        "//labm8:app",
        "//third_party/py/pandas",
    ],
)

filegroup(
    name = "npb_sources_tar",
    srcs = ["npb-3.3.tar.bz2"],
)
