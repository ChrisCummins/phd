py_binary(
    name = "github_c",
    srcs = ["github_c.py"],
    deps = [
        "//compilers/llvm:clang",
        "//datasets/github/scrape_repos:contentfiles",
        "//experimental/compilers/reachability:database",
        "//experimental/compilers/reachability:reachability_py_pb2",
        "//labm8:lockfile",
        "//third_party/py/absl",
        "//third_party/py/humanize",
        "//third_party/py/sqlalchemy",
    ],
)
