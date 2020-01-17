<h1>
  gh-archiver
  </a> <a href="https://tldrlegal.com/license/mit-license" target="_blank">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat">
  </a>
</h1>

Mirror a GitHub user's repos locally.

This program fetches a GitHub user's repositories and mirrors them to a local
directory. New repositories are cloned, existing repositories are fetched from
remote.

## Setup

Create a Github [personal access token](https://github.com/settings/tokens/new).
If you intend to mirror your own private repositories, select "repo" from the
list of available scopes. To mirror only your public repositories or those
another user, no scopes are required.

Create a ~/.github/access_tokens/gh_archiver.txt file containing your
the personal access token you just created:

```sh
$ mkdir -p ~/.github/access_tokens
$ cat <<EOF > ~/.github/access_tokens/gh_archiver.txt
YourAccessToken
EOF
$ chmod 0600 ~/.github/access_tokens/gh_archiver.txt
```

Then build and install the `gh_archiver` program using:

```sh
$ basel run -c opt //datasets/github/gh_archiver:install
```

Requires Python >= 3.6.

## Usage

Mirror a Github user's repositories to a directory using:

```sh
$ gh_archiver --user <github_username> --outdir <path>
```


## License

Made with ❤️ by [Chris Cummins](http://chriscummins.cc).
Released under [MIT License](https://tldrlegal.com/license/mit-license).
