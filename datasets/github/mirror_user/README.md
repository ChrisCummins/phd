# Mirror a GitHub user's repos 

This package can be used to mirror a GitHub user's repositories locally. I use
it to keep a "backup" of my GitHub repositories automatically synced to a local
NAS server running [gogs](https://gogs.io).

## Pre-requisites

Create a file `~/.githubrc`:

```ini
[User]
Username = your-github-username
Password = your-github-password
```

## Mirror repositories locally

Clone and update a GitHub user's repositories locally:

```sh
$ bazel run //datasets/github/mirror_user -- \
    --user=ChrisCummins --dst=$HOME/src/GitHub/
```

## Mirror to Gogs serer

Create a credentials file `~/.gogsrc` with your Gogs server address and 
[token](https://github.com/gogs/docs-api#access-token):

```sh
[Server]
Address = http://example.com:3000

[User]
Token = 39bbdb529fed7fc4f373410518745446d9901450
```

Clone and update a GitHub user's repositories on a [gogs](https://gogs.io) 
server:

```sh
$ bazel run //datasets/github/mirror_user -- \
    --user=ChrisCummins --dst=$HOME/gogs/repos/ChrisCummins --gogs --gogs-uid 1
```

Alternatively, use flag `--gogsrc=/path/to/gogsrc` to specify the credentials 
file path.

## License

Made with ❤️ by [Chris Cummins](http://chriscummins.cc). Released under 
[MIT License](https://tldrlegal.com/license/mit-license).
