# End-to-end integration test.

py_test(
    name = "java_fuzz_integration_test",
    srcs = ["java_fuzz_integration_test.py"],
    data = [
        "//experimental/deeplearning/deepsmith/java_fuzz:export_java_corpus_image.tar",
        "//experimental/deeplearning/deepsmith/java_fuzz:mask_contentfiles_image.tar",
        "//experimental/deeplearning/deepsmith/java_fuzz:scrape_java_files_image.tar",
        "//labm8:dockerutil",
        "//labm8:test",
    ],
)
