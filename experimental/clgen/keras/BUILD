# Experiments while porting CLgen from Tensorflow to Keras models.

py_test(
    name = "benchmark_module_import_test",
    srcs = ["benchmark_module_import_test.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/keras",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "benchmark_model_init_test",
    srcs = ["benchmark_model_init_test.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/keras",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "benchmark_string_concatenation_test",
    srcs = ["benchmark_string_concatenation_test.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:test",
    ],
)

py_test(
    name = "log_to_file_test",
    srcs = ["log_to_file_test.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:logutil",
        "//labm8/py:test",
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
        "//deeplearning/clgen:sample_observers",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
    ],
)
