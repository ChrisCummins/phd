# Connecting to GitHub's API

This package defines the unified method for authenticating connections to
Github's API. Tools which use this package all authenticate using
[access tokens](https://github.com/settings/tokens). See the documentation for
the specific tool to see what scopes are expected of a token.

Tools which use this package resolve access tokens using the following order:

1. If the tool provides a list of paths that it expects to find access tokens
   in, first try and read a token from those.
1. If `$GITHUB_ACCESS_TOKEN` is set, use it.
1. If `$GITHUB_ACCESS_TOKEN_PATH` is set and points to a file, attempt to read
   it. If the variable is set but the file does not exist, a warning is printed.
1. If `--github_access_token` is set, use it.
1. If `--github_credentials_path` points to a file, use it.
1. If we are in a test environment and `~/.github/access_tokens/test.txt` is a
   file, read the token from that.
