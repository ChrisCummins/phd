# Clone Popular GitHub Repos by Language

This package implements a two-step process to clone the *n* most popular 
repositories of a particular programming language from GitHub.

## Pre-requisites

Create a file `~/.githubrc`:

```ini
[User]
Username = your-github-username
Password = your-github-password
```


## Usage

Create a "clone list" of languages and repositories to scrape:

```sh
$ cat ./clone_list.pbtxt
# File: //datasets/github/scrape_repos/proto/scrape_repos.proto
# Proto: scrape_repos.LanguageCloneList

language {
  language: "java"
  destination_directory: "/tmp/java"
  query {
    string: "language:java sort:stars fork:false"
    max_results: 10
  }
}
```

Build the project:

```sh
$ bazel build //datasets/github/scrape_repos/...
```

Scrape GitHub to create `GitHubRepositoryMeta` messages of repos using:

```sh
$ $PHD/bazel-out/*-py3-opt/bin/datasets/github/scrape_repos/scraper \
    --clone_list=$PWD/clone_list.pbtxt
```

Run the cloner to download the repos scraped in the previous step:

```sh
$ $PHD/bazel-out/*-py3-opt/bin/datasets/github/scrape_repos/cloner \
    --clone_list=$PWD/clone_list.pbtxt
```

Run the importer to put source files into contentfiles databases:

```sh
$ $PHD/bazel-out/*-py3-opt/bin/datasets/github/scrape_repos/importer \
    --clone_list=$PWD/clone_list.pbtxt
``` 

Export the corpus to a directory:

```sh
$ $PHD/bazel-out/*-py3-opt/bin/datasets/github/scrape_repos/export_corpus \
           --clone_list=$PWD/clone_list.pbtxt --export_path=/tmp/java_corpus
```
