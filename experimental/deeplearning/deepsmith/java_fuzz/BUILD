load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_binary(
    name = "encode_java_corpus",
    srcs = ["encode_java_corpus.py"],
    data = ["//deeplearning/clgen/corpuses/lexer:lexer_worker"],
    deps = [
        "//deeplearning/clgen/corpuses:encoded",
        "//deeplearning/clgen/corpuses:preprocessed",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:pbutil",
        "//labm8:sqlutil",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "encode_java_corpus_test",
    srcs = ["encode_java_corpus_test.py"],
    deps = [
        ":encode_java_corpus",
        "//deeplearning/clgen/corpuses:encoded",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py3_image(
    name = "encode_java_corpus_image",
    srcs = ["encode_java_corpus.py"],
    base = "@base_java//image",
    main = "encode_java_corpus.py",
    visibility = ["//experimental/deeplearning/deepsmith/java_fuzz:__subpackages__"],
    deps = [":encode_java_corpus"],
)

py_binary(
    name = "export_java_corpus",
    srcs = ["export_java_corpus.py"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//datasets/github/scrape_repos/preprocessors:extractors",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:humanize",
        "//labm8:sqlutil",
    ],
)

py3_image(
    name = "export_java_corpus_image",
    srcs = ["export_java_corpus.py"],
    base = "@base//image",
    main = "export_java_corpus.py",
    visibility = ["//experimental/deeplearning/deepsmith/java_fuzz:__subpackages__"],
    deps = [":export_java_corpus"],
)

py_test(
    name = "export_java_corpus_test",
    srcs = ["export_java_corpus_test.py"],
    deps = [
        ":export_java_corpus",
        "//datasets/github/scrape_repos:contentfiles",
        "//labm8:test",
        "//third_party/py/pytest",
    ],
)

py_binary(
    name = "export_preprocessed_methods",
    srcs = ["export_preprocessed_methods.py"],
    deps = [
        "//deeplearning/clgen/corpuses:preprocessed",
        "//labm8:app",
        "//labm8:fs",
        "//third_party/py/progressbar",
    ],
)

py_binary(
    name = "export_rewriter_failures",
    srcs = ["export_rewriter_failures.py"],
    deps = [
        ":preprocess_java_corpus",
        "//datasets/github/scrape_repos:contentfiles",
        "//datasets/github/scrape_repos/proto:scrape_repos_pb_py",
        "//deeplearning/clgen/corpuses:preprocessed",
        "//deeplearning/clgen/preprocessors:java",
        "//labm8:app",
        "//labm8:fs",
        "//labm8:humanize",
        "//third_party/py/sqlalchemy",
    ],
)

py_binary(
    name = "mask_contentfiles",
    srcs = ["mask_contentfiles.py"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//labm8:app",
        "//labm8:humanize",
        "//third_party/py/sqlalchemy",
    ],
)

py3_image(
    name = "mask_contentfiles_image",
    srcs = ["mask_contentfiles.py"],
    base = "@base//image",
    main = "mask_contentfiles.py",
    visibility = ["//experimental/deeplearning/deepsmith/java_fuzz:__subpackages__"],
    deps = [":mask_contentfiles"],
)

py_binary(
    name = "scrape_java_files",
    srcs = ["scrape_java_files.py"],
    deps = [
        "//datasets/github:api",
        "//datasets/github/scrape_repos:contentfiles",
        "//datasets/github/scrape_repos:pipelined_scraper",
        "//datasets/github/scrape_repos/proto:scrape_repos_pb_py",
        "//labm8:app",
        "//third_party/py/github",
        "//third_party/py/progressbar",
        "//third_party/py/urllib3",
    ],
)

py3_image(
    name = "scrape_java_files_image",
    srcs = ["scrape_java_files.py"],
    base = "@base//image",
    main = "scrape_java_files.py",
    visibility = ["//experimental/deeplearning/deepsmith/java_fuzz:__subpackages__"],
    deps = [":scrape_java_files"],
)

py_binary(
    name = "preprocess_java_corpus",
    srcs = ["preprocess_java_corpus.py"],
    data = ["//deeplearning/clgen/preprocessors:JavaPreprocessor"],
    deps = [
        "//datasets/github/scrape_repos:contentfiles",
        "//datasets/github/scrape_repos/preprocessors:secrets",
        "//datasets/github/scrape_repos/proto:scrape_repos_pb_py",
        "//deeplearning/clgen/corpuses:preprocessed",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:pbutil",
    ],
)

py3_image(
    name = "preprocess_java_corpus_image",
    srcs = ["preprocess_java_corpus.py"],
    base = "@base_java//image",
    main = "preprocess_java_corpus.py",
    visibility = ["//experimental/deeplearning/deepsmith/java_fuzz:__subpackages__"],
    deps = [":preprocess_java_corpus"],
)

py_test(
    name = "preprocess_java_corpus_test",
    srcs = ["preprocess_java_corpus_test.py"],
    deps = [
        ":preprocess_java_corpus",
        "//datasets/github/scrape_repos/proto:scrape_repos_pb_py",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8:test",
    ],
)

py_binary(
    name = "re_preprocess_java_methods",
    srcs = ["re_preprocess_java_methods.py"],
    deps = [
        ":preprocess_java_corpus",
        "//datasets/github/scrape_repos:contentfiles",
        "//datasets/github/scrape_repos/preprocessors:secrets",
        "//datasets/github/scrape_repos/proto:scrape_repos_pb_py",
        "//deeplearning/clgen/corpuses:preprocessed",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8:app",
        "//labm8:bazelutil",
        "//labm8:pbutil",
    ],
)

py3_image(
    name = "re_preprocess_java_methods_image",
    srcs = ["re_preprocess_java_methods.py"],
    base = "@base_java//image",
    main = "re_preprocess_java_methods.py",
    visibility = ["//experimental/deeplearning/deepsmith/java_fuzz:__subpackages__"],
    deps = [":re_preprocess_java_methods"],
)

py_binary(
    name = "sample_java_model",
    srcs = ["sample_java_model.py"],
    visibility = ["//experimental/deeplearning/deepsmith/java_fuzz:__subpackages__"],
    deps = [
        "//deeplearning/clgen",
        "//deeplearning/clgen:samples_database",
        "//deeplearning/clgen/corpuses:encoded",
        "//deeplearning/clgen/proto:clgen_pb_py",
        "//labm8:app",
        "//labm8:crypto",
        "//labm8:pbutil",
    ],
)

py3_image(
    name = "sample_java_model_image",
    srcs = ["sample_java_model.py"],
    base = "@base_java//image",
    main = "sample_java_model.py",
    visibility = ["//experimental/deeplearning/deepsmith/java_fuzz:__subpackages__"],
    deps = [":sample_java_model"],
)

filegroup(
    name = "shared_code_README",
    srcs = ["shared_code_README.md"],
)

py_binary(
    name = "EXPORT",
    srcs = ["EXPORT.py"],
    deps = ["//tools/source_tree:export_source_tree"],
)
