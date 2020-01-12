# Format: Automated Code Formatters

This package implements an opinionated, non-configurable enforcer of code style.

```sh
$ format <path ...>
```

This program enforces a consistent code style on files by modifying them in
place. If a path is a directory, all files inside it are formatted.

Features:

  * Automated code styling of C/C++, Python, Java, SQL, JavaScript, HTML,
    CSS, go, and JSON files.
  * Support for `.formatignore` files to mark files to be excluded from 
    formatting. The syntax of ignore files is similar to `.gitignore`, e.g. a 
    list of patterns to match, including (recursive) glob expansion, and 
    patterns beginning with `!` are un-ignored. 
  * Persistent caching of "last modified" timestamps for files to minimize the
    amount of work done.
  * A process lock which prevents races when multiple formatters are launched
    simultaneously.

The type of formatting applied to a file is determined by its suffix. See
format --print_suffixes for a list of suffixes which are formatted.

This program uses a filesystem cache to store various attributes such as a
database of file modified times. See `format --print_cache_path` to print the
path of the cache. Included in the cache is a file lock which prevents mulitple
instances of this program from modifying files at the same time, irrespective
of the files being formatted.

## Setup

### Requirements

1. Python >= 3.6.
1. sqlite

### Install

Download the latest binary from the [Releases page](https://github.com/ChrisCummins/format) and put it in your $PATH.

Or to build from source:

```sh
$ bazel run -c opt //tools/format:install
```

## License

Copyright 2020 Chris Cummins <chrisc.101@gmail.com>.

Released under the terms of the Apache 2.0 license. See
`LICENSE` for details.
