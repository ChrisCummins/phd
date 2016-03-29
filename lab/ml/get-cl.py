#!/usr/bin/env python3
#
# Download all of the OpenCL files on GitHub (!)
#
import json
import os
import sys

import requests

from base64 import b64decode
from github import Github


def main():
    try:
        github_username = os.environ['GITHUB_USERNAME']
        github_pw = os.environ['GITHUB_PW']
        github_token = os.environ['GITHUB_TOKEN']
    except KeyError as e:
        print('fatal: environment variable {} not set'.format(e))
        sys.exit(1)

    g = Github(github_username, github_pw)

    # Fetch the repositories to iterate over. Since opencl isn't
    # treated as a first-class language by GitHub, we can't use the
    # 'language=' keyword for queries, so instead just look for the
    # keyword 'opencl'.
    #
    query = g.search_repositories('opencl stars:<4', sort='stars')
    # query = g.search_repositories('cuda', sort='stars')
    # query = g.search_repositories('opencl language:c++', sort='stars')
    repos = []

    for repo in query:
        print(repo.url, repo.stargazers_count, repo.default_branch)

        # Iterate over the entire git tree of the repo's default
        # branch (usually 'master'). If a file ends with the .cl
        # extension, check to see if we already have it, else download
        # it.
        #
        for f in repo.get_git_tree(repo.default_branch, recursive=True).tree:
            if f.path.endswith('.cl'):
                print('  ', f.path)
                # File name template: 'cl/<repo>--<path>', where
                # <repo> is the repository name and <path> is the file
                # path, with path separators ('/') replaced with
                # hyphens ('-').
                #
                local_name = repo.name + '--' + f.path.replace('/', '-')
                local_file = 'cl/' + local_name

                if not os.path.exists(local_file):
                    response = json.loads(requests.get(
                        f.url,
                        headers = {'Authorization': 'token ' + str(github_token)}
                    ).content.decode('utf-8'))
                    code = b64decode(response['content'])

                    try:
                        with open(local_file, 'bw') as outfile:
                            outfile.write(code)
                    except Exception as e:
                        print(code)
                        sys.exit(1)

        print()


if __name__ == '__main__':
    main()
