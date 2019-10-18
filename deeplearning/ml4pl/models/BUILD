# Machine learning models.

py_library(
    name = "log_writer",
    srcs = ["log_writer.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//labm8:app",
        "//labm8:jsonutil",
    ],
)

py_test(
    name = "log_writer_test",
    srcs = ["log_writer_test.py"],
    deps = [
        ":log_writer",
        "//labm8:app",
        "//labm8:test",
    ],
)

py_library(
    name = "batch_logger",
    srcs = ["batch_logger.py"],
    visibility = ["//deeplearning/ml4pl:__subpackages__"],
    deps = [
        "//labm8:app",
    ],
)

py_test(
    name = "batch_logger_test",
    srcs = ["batch_logger_test.py"],
    deps = [
        ":batch_logger",
        "//labm8:app",
        "//labm8:test",
    ],
)
