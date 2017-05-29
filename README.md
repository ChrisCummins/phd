<h1>
  gh-archiver
  <a href="https://badge.fury.io/py/gh-archiver">
    <img src="https://img.shields.io/pypi/v/gh-archiver.svg?colorB=green&style=flat">
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

## Installation

```sh
$ pip install gh-archiver
```

Requires Python >= 3.6.

## Usage

Create a credentials file `~/.githubrc` with your GitHub username and password:

```sh
$ cat <<EOF > ~/.githubrc
[User]
Username = YourUsername
Password = password1234
EOF
$ chmod 0600 ~/.githubrc
```

Alternatively, use flag `--githubrc <path>` to specify an alternate path to the credentials file.

## License

Made with ❤️ by [Chris Cummins](http://chriscummins.cc). Released under [MIT License](https://tldrlegal.com/license/mit-license).
