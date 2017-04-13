cc_binary(
    name =  "example1",
    srcs = ["example1.cc"],
    copts = [
        "-Iplayground/rt/include",
        "-Iexternal/tbb/include",
        "-Iexternal/tbb/src/rml/include",
    ],
    deps = [
        "//playground/rt:main",
        "@tbb//:main",
    ],
)
