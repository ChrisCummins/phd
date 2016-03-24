#!/usr/bin/env python3
#
# Challenge: Plot the distribution of commit message lengths in each
# of my public GitHub repos.
#
import os
import sys

from github import Github


def get_github_token(username, password):
    import json
    import requests
    response = requests.post(
        'https://api.github.com/authorizations',
        auth = (username, password),
        data = json.dumps({ 'scopes': ['repo'], 'note': 'phd auth' }),
    )
    return response.json()['token']


def main():
    try:
        github_username = os.environ['GITHUB_USERNAME']
        github_pw = os.environ['GITHUB_PW']
        # github_token = get_github_token(github_username, github_pw)
    except KeyError as e:
        print('fatal: environment variable {} not set'.format(e))
        sys.exit(1)

    g = Github(github_username, github_pw)

    # Get sorted list of public, non-forked repos
    repos = [x for x in g.get_user().get_repos()
             if not x.fork and not x.private]
    repos.sort(key=lambda x: x.name.lower())

    repo_count = 0
    for repo in repos:
        lc = 0
        wc = 0
        commit_count = 0
        for commit in repo.get_commits():
            # Only count commits that I wrote
            if commit.commit.author.email == github_username:
                message = commit.commit.message
                lc += len(message.split('\n'))
                wc += len(message.split())
                commit_count += 1

        # Check that there was at least one commit:
        if commit_count:
            lc_mean = lc / commit_count * 1.0
            wc_mean = wc / commit_count * 1.0

            print(repo.name, commit_count, round(lc_mean), round(wc_mean))
            repo_count += 1

    print(repo_count, 'repos')


if __name__ == '__main__':
    main()
