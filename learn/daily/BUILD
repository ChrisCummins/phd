# Daily programming challenges.

cc_library(
    name = "181211_matrix_multiplication",
    srcs = ["181211_matrix_multiplication.cc"],
    hdrs = ["181211_matrix_multiplication.h"],
    deps = [
        "//phd:macros",
        "@boost//:numeric",
        "@boost//:thread",
    ],
)

cc_test(
    name = "181211_matrix_multiplication_test",
    srcs = ["181211_matrix_multiplication_test.cc"],
    deps = [
        ":181211_matrix_multiplication",
        "//phd:test",
    ],
)
