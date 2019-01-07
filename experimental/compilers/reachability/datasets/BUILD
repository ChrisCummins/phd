py_binary(
    name = "github",
    srcs = ["github.py"],
    deps = [
        "//compilers/llvm:clang",
        "//datasets/github/scrape_repos:contentfiles",
        "//deeplearning/clgen/preprocessors:opencl",
        "//experimental/compilers/reachability:database",
        "//experimental/compilers/reachability:reachability_py_pb2",
        "//labm8:lockfile",
        "//labm8:ppar",
        "//third_party/py/absl",
        "//third_party/py/sqlalchemy",
    ],
)
