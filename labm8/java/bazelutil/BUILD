# Classes for working within Bazel's hemetic and sandboxed world.

java_library(
    name = "BazelRunfiles",
    srcs = ["BazelRunfiles.java"],
    visibility = ["//visibility:public"],
    deps = [
        "@bazel_tools//tools/java/runfiles",
    ],
)

java_test(
    name = "BazelRunfilesTest",
    srcs = ["BazelRunfilesTest.java"],
    data = ["//labm8/test_data:hello_world_file"],
    test_class = "labm8.java.bazelutil.BazelRunfilesTest",
    deps = [
        ":BazelRunfiles",
        "@org_junit//jar",
    ],
)
