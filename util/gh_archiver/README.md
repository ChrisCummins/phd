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

Create a Github [personal access token](https://github.com/settings/tokens). If
you intent to mirror your own private repositories, you should enable private
repository permissions when creating the token. Else, no permissions are
required.

Create a ~/.githubrc file containing your Github username and the personal
access token you just created::

```sh
$ cat <<EOF > ~/.githubrc
[User]
Username = YourUsername

[Tokens]
gh_archiver = YourAccessToken
EOF
$ chmod 0600 ~/.githubrc
```

Then build and install the `gh_archiver` program using:

```sh
$ basel run -c opt //util/gh_archiver:install
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
