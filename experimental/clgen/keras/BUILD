# Experiments while porting CLgen from Tensorflow to Keras models.

py_test(
    name = "benchmark_module_import_test",
    srcs = ["benchmark_module_import_test.py"],
    deps = [
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/keras",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "benchmark_model_init_test",
    srcs = ["benchmark_model_init_test.py"],
    deps = [
        "//labm8:app",
        "//labm8:test",
        "//third_party/py/keras",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "benchmark_string_concatenation_test",
    srcs = ["benchmark_string_concatenation_test.py"],
    deps = [
        "//labm8:app",
        "//labm8:test",
    ],
)

py_test(
    name = "log_to_file_test",
    srcs = ["log_to_file_test.py"],
    deps = [
        "//labm8:app",
        "//labm8:logutil",
        "//labm8:test",
    ],
)

py_binary(
    name = "opencl_baseline",
    srcs = ["opencl_baseline.py"],
    data = [
        "opencl_baseline_a.pbtxt",
        "opencl_baseline_b.pbtxt",
    ],
    deps = [
        "//deeplearning/clgen",
        "//labm8:app",
        "//labm8:bazelutil",
    ],
)
