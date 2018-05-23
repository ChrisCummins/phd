# Clone Popular GitHub Repos by Language

This package implements a two-step process to clone the *n* most popular 
repositories of a particular programming language from GitHub.


## Usage

Create a "clone list" of languages and repositories to scrape:

```bash
$ cat ./clone_list.pbtxt
# File: //datasets/github/scrape_repos/proto/scrape_repos.proto
# Proto: scrape_repos.LanguageCloneList
language {
  language: "javascript"
  num_repos_to_clone: 100
  destination_directory: '/var/github_repos/javascript'
}
language {
  language: "c"
  num_repos_to_clone: 500
  destination_directory: '/var/github_repos/c'
}
```

Scrape GitHub to create `GitHubRepositoryMeta` messages of repos using:

```sh
$ bazel run //datasets/github/scrape_repos:scraper -- \
    --clone_list $PWD/clone_list.pbtxt
```

Rune the cloner to download the repos scraped in the previous step:

```sh
$ bazel run //datasets/github/scrape_repos:cloner -- \
    --clone_list $PWD/clone_list.pbtxt
```
