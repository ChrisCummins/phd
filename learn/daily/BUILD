# Daily programming challenges.

# Targets in this package are prefixed by the letter "d" followed by their date
# in YYMMDD format. The "d" prefix is to ensure that python module names are
# legal.

cc_library(
    name = "d181211_matrix_multiplication",
    srcs = ["d181211_matrix_multiplication.cc"],
    hdrs = ["d181211_matrix_multiplication.h"],
    deps = [
        "//phd:macros",
        "@boost//:numeric",
        "@boost//:thread",
    ],
)

cc_test(
    name = "d181211_matrix_multiplication_test",
    srcs = ["d181211_matrix_multiplication_test.cc"],
    deps = [
        ":d181211_matrix_multiplication",
        "//phd:test",
    ],
)
