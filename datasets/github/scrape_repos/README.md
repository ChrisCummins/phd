# Clone Popular GitHub Repos by Language

This package provides a means to scrape repositories and source code files from
GitHub.

## Pre-requisites

Create a file `~/.githubrc`:

```ini
[User]
Username = your-github-username
Password = your-github-password
```


## Usage

Create a "clone list" file which contains a list of languages and the
[GitHub repository queries](https://help.github.com/articles/searching-repositories/)
to run for each:

```sh
$ cat ./clone_list.pbtxt
# File: //datasets/github/scrape_repos/proto/scrape_repos.proto
# Proto: scrape_repos.LanguageCloneList

language {
  language: "java"
  destination_directory: "/tmp/phd/datasets/github/scrape_repos/java"
  query {
    string: "language:java sort:stars fork:false"
    max_results: 10
  }
}
```

See schema defined in
[//datasets/github/scrape_repos/proto/scrape_repos.proto](/datasets/github/scrape_repos/proto/scrape_repos.proto).

Scrape GitHub to create `GitHubRepositoryMeta` messages of repos using:

```sh
$ bazel run //datasets/github/scrape_repos/scraper \
    --clone_list $PWD/clone_list.pbtxt
```

Run the cloner to download the repos scraped in the previous step:

```sh
$ bazel run //datasets/github/scrape_repos/cloner \
    --clone_list $PWD/clone_list.pbtxt
```

Extract individual source files from the cloned repos and import them into a
[contentfiles database](/datasets/github/scrape_repos/contentfiles.py) using:

```sh
$ bazel run //datasets/github/scrape_repos/importer \
    --clone_list $PWD/clone_list.pbtxt
```

Export the source files from the corpus database to a directory:

```sh
$ bazel run //datasets/github/scrape_repos/export_corpus \
    --clone_list $PWD/clone_list.pbtxt \
    --export_path /tmp/phd/datasets/github/scrape_repos/corpuses/java
```
