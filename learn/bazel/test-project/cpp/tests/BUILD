cc_test(
    name = "hello-test",
    srcs = ["hello-test.cc"],
    copts = [
        "-Iexternal/gtest/include",
        "-Icpp"
    ],
    deps = [
        "@gtest//:main",
        "//cpp/lib:hello-greet",
    ],
)
