cc_test(
    name = "001-fib",
    size = "small",
    srcs = ["001-fib.cc"],
    deps = [
        "@com_github_google_benchmark//:benchmark",
        "@gtest",
    ],
)

cc_test(
    name = "002-linked-lists",
    size = "small",
    srcs = ["002-linked-lists.cc"],
    deps = [
        "@com_github_google_benchmark//:benchmark",
        "@gtest",
    ],
)

cc_test(
    name = "003-string-inc",
    size = "small",
    srcs = ["003-string-inc.cc"],
    copts = [
        "-Ilib/phd/include",
    ],
    deps = ["//lib/phd:main"],
)

cc_test(
    name = "004-fractal",
    size = "small",
    srcs = ["004-fractal.cc"],
    copts = [
        "-Ilib/phd/include",
    ],
    deps = ["//lib/phd:main"],
)

cc_binary(
    name = "humble-001",
    srcs = ["humble-001.cc"],
    args = [
        "-f",
        "1",
        "-g",
        "2",
    ],
)

cc_binary(
    name = "humble-003-vector",
    srcs = ["humble-003-vector.cc"],
)

cc_test(
    name = "add_up_to_k",
    srcs = ["add_up_to_k.cc"],
    deps = ["//labm8/cpp:test"],
)

py_test(
    name = "binary_search_second_largest_test",
    srcs = ["binary_search_second_largest_test.py"],
    deps = [
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py_test(
    name = "cons_car_cdr_test",
    srcs = ["cons_car_cdr_test.py"],
    deps = [
        "//labm8:test",
    ],
)

py_test(
    name = "serialize_binary_tree",
    srcs = ["serialize_binary_tree.py"],
)

cc_test(
    name = "vector_of_products",
    srcs = ["vector_of_products.cc"],
    deps = ["//labm8/cpp:test"],
)
