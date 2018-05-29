# Protos for mining programming language datasets from GitHub.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "scrape_repos_py_pb2",
    protos = ["scrape_repos.proto"],
    visibility = [
        "//datasets/github/scrape_repos:__subpackages__",
        "//experimental:__subpackages__",
    ],
)
