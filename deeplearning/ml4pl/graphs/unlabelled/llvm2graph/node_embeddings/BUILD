# Agumented inst2vec embeddings.
#
# This extends the vocabulary used in the NeurIPS'18 paper to add three extra
# elements: !MAGIC, !IMMEDIATE, and !IDENTIFIER. These new vocabulary elements
# have been assigned random embedding vectors.

filegroup(
    name = "dictionary",
    srcs = ["inst2vec_augmented_dictionary.pickle"],
    visibility = ["//deeplearning/ml4pl/graphs/unlabelled/llvm2graph:__subpackages__"],
)

filegroup(
    name = "embeddings",
    srcs = ["inst2vec_augmented_embeddings.pickle"],
    visibility = ["//deeplearning/ml4pl/graphs/unlabelled/llvm2graph:__subpackages__"],
)
