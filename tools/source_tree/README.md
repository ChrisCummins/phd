# Export bazel subtree to GitHub

This package contains a utility for exporting a subset of my
[phd](https://github.com/ChrisCummins/phd) repo as a standalone git repository.
Give one or more bazel targets, it determines the required dependencies and
filters through the git history, exporting only the commits that are needed
to reproduce the files, and rewriting the commits so that only the necessary
files are modified.

## Usage

To export the targets `//package/to/export/...` and `//another/package` to a
GitHub repo called `github_export`:

```sh
$ bazel run //tools/source_tree:export_git_history -- \
        --targets=//package/to/export/...,//another/package \
        --mv_files=package/to/export/README.md:README.md \
        --github_repo=github_export
```

Additionally, the `--mv_files` argument permits moving a file's location in
the exported repository, which is useful for exporting things like readme and
license files to the package root.

The utility can also be called from a python script. The script equivalent to
the previous command is:

```py
from tools.source_tree import export_source_tree
export_source_tree.EXPORT(
    github_repo='git_bazel_subtree_filter',
    targets=[
        '//package/to/export/...',
        '//another/package',
    ],
    move_file_mapping={
        'package/to/export/README.md': 'README.md',
    },
)
```

And the `BUILD` target for it:

```
py_binary(
    name = "EXPORT",
    srcs = ["EXPORT.py"],
    deps = ["//tools/source_tree:export_source_tree"],
)
```
