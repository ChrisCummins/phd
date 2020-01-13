# Format: Automated Code Formatter

This projects implements an opinionated, non-configurable enforcer of code
style. The aim is to take control of source formatting away from the developer,
reducing cognitive load and allowing you to focus on what matters.

## Features

☑️ **Consistent** code styling of C/C++, Python, Java, SQL, JavaScript, HTML,
  CSS, Go, Markdown, plain text, and JSON files.

☑️ **Git-aware** pre-commit mode which formats changed files and signs off
  commits.

☑️ **Fast** incremental formats of large code bases using a "last modified"
  time stamp cache.

☑️ **Black-listing** of files from automated formatting using git-like ignore
  files.

☑️ **Safe** execution using inter-process locking to prevent multiple
  formatters modifying files simultaneously.

## Install

Download a binary from the
[release page](https://github.com/ChrisCummins/format/releases) and put it in
your `$PATH`.

Alternatively you can build and install the formatter from source in this
repository using:

```sh
$ git clone https://github.com/ChrisCummins/format.git
$ cd format
$ bazel run -c opt //tools/format:install
```

Requires Python >= 3.6 and sqlite. Syntax-specific formatters may have
additional dependencies:

* **Java** files require a host java.
* **JSON** files require [jsonlint](https://www.npmjs.com/package/jsonlint).

## Usage

Format files in place using:

```sh
$ format <path ...>
```

If a path is a directory, all files inside it are formatted. The type of
formatting applied to a file is determined by its suffix. To print the files
that will be formatted without changing them, use:

```sh
$ format --dry_run <path ...>
```

If you want to exclude files from formatting, create a `.formatignore` file. It
has similar rules to `.gitignore` files, e.g.:

```sh
$ cat .formatignore
# Everything after '#' character is a comment
hello.txt  # Name specific files to exclude from formatting
**/*.json  # Or glob them
!package.json  # Use '!' character to create un-ignore patterns
```

For git repositories, you can install the formatter to execute as a pre-commit
hook using:

```sh
$ format --install_pre_commit_hook
```

This installs hooks that run the formatter on changelists before commits, and
adds a "Signed-off-by" footer to commit messages verifying that the commit
contents passed formatting checks.

This program uses a filesystem cache to store various attributes such as a
database of file modified times. See `format --print_cache_path` to print the
path of the cache.

## License

Copyright 2020 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the Apache 2.0 license. See `LICENSE` for details.