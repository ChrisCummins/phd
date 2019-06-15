# Connecting to GitHub's API

There are three options for authenticating a connection to the Github API:

1. Create a [personal access token](https://github.com/settings/tokens) and 
   pass it as a command line argument `--github_access_token=<token>`.
2. Create a [personal access token](https://github.com/settings/tokens), put in
   a file and pass the path of the file as a command line argument 
   `--github_access_token_path=/path/to/token`.
3. (not recommended) Create a file `~/.githubrc` and put your GitHub username 
   and password in it:

```ini
[User]
Username = your-github-username
Password = your-github-password
```
