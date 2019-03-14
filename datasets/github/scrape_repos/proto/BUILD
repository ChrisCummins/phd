# Protos for mining programming language datasets from GitHub.
#
# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")
load("@org_pubref_rules_protobuf//java:rules.bzl", "java_proto_library")

py_proto_library(
    name = "scrape_repos_py_pb2",
    protos = ["scrape_repos.proto"],
    visibility = [
        "//datasets/github/scrape_repos:__subpackages__",
        "//experimental:__subpackages__",
    ],
)

java_proto_library(
    name = "scrape_repos_java",
    protos = ["scrape_repos.proto"],
    visibility = [
        "//datasets/github/scrape_repos:__subpackages__",
        "//experimental:__subpackages__",
    ],
)
