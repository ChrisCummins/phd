# Test data for CLgen.

filegroup(
    name = "cl_kernels",
    srcs = glob([
        "cl/*.cl",
        "cl/*.h",
        "cl/*.gs",
    ]),
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

filegroup(
    name = "dbs",
    srcs = glob(["db/*.db"]),
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

filegroup(
    name = "pico",
    srcs = [
        "pico/corpus.tar.bz2",
        "pico/model.json",
        "pico/sampler.json",
    ],
    visibility = ["//deeplearning/clgen:__subpackages__"],
)

filegroup(
    name = "tiny",
    srcs = [
        "tiny/corpus.tar.bz2",
        "tiny/model.json",
        "tiny/sampler.json",
    ],
    visibility = ["//deeplearning/clgen:__subpackages__"],
)
