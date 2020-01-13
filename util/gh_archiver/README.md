<h1>
  gh-archiver
  </a> <a href="https://tldrlegal.com/license/mit-license" target="_blank">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat">
  </a>
</h1>

Clone and update a GitHub user's repositories locally:

```sh
$ gh-archiver ChrisCummins -o ~/src/GitHub/
cloning atom
cloning autoencoder
updating chriscummins.cc
...
```

Or mirror to a [gogs](https://gogs.io) server:

```sh
$ gh-archiver ChrisCummins -o ~/gogs/repos/ChrisCummins --gogs --gogs-uid 1
mirring atom ... 201
mirroring autoencoder ... 201
mirroring chriscummins.cc ... 201
...
```

## Installation

```sh
$ bazel run //util/gh_archiver:install
```

Requires Python >= 3.6.

**GitHub credentials**

Create a credentials file `~/.githubrc` with your GitHub username and a
[personal access token](https://github.com/settings/tokens):

```sh
$ cat <<EOF > ~/.githubrc
[User]
Username = YourUsername

[Tokens]
gh_archiver = YourAccessToken
EOF
$ chmod 0600 ~/.githubrc
```

Alternatively, use flag `--githubrc <path>` to specify a path to the credentials file.

**Gogs credentials**

Create a credentials file `~/.gogsrc` with your Gogs server address and [token](https://github.com/gogits/go-gogs-client/wiki#access-token):

```sh
$ cat <<EOF > ~/.gogsrc
[Server]
Address = http://example.com:3000

[User]
Token = YourToken
EOF
$ chmod 0600 ~/.gogsrc
```

Alternatively, use flag `--gogsrc <path>` to specify a path to the credentials file.

## License

Made with ❤️ by [Chris Cummins](http://chriscummins.cc). Released under [MIT License](https://tldrlegal.com/license/mit-license).
