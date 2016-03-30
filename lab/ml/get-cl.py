#!/usr/bin/env python3
#
# Download all of the OpenCL files on GitHub (!)
#
# TODO: Check for .ocl files perhaps?
#
import json
import os
import sys
import sqlite3

import requests

from base64 import b64decode
from github import Github,GithubException


def main():
    try:
        github_username = os.environ['GITHUB_USERNAME']
        github_pw = os.environ['GITHUB_PW']
        github_token = os.environ['GITHUB_TOKEN']
    except KeyError as e:
        print('fatal: environment variable {} not set'.format(e))
        sys.exit(1)

    g = Github(github_username, github_pw)
    conn = sqlite3.connect('cl.db')

    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS repos "
              "(url TEXT NOT NULL,"
              " author TEXT NOT NULL,"
              " name TEXT NOT NULL,"
              " stars INT NOT NULL, "
              " fork INT NOR NULL, "
              " updated_at DATETIME NOT NULL, "
              "UNIQUE(url))")
    c.execute("CREATE TABLE IF NOT EXISTS files "
              "(url TEXT NOT NULL, "
              " path TEXT NOT NULL, "
              " repo_id TEXT NOT NULL, "
              " source TEXT NOT NULL, "
              " date_added DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, "
              "UNIQUE(url))")
    conn.commit()

    # Fetch the repositories to iterate over. Since opencl isn't
    # treated as a first-class language by GitHub, we can't use the
    # 'language=' keyword for queries, so instead just look for the
    # keyword 'opencl'.
    #
    query = g.search_repositories('opencl', sort='stars')
    # query = g.search_repositories('cuda', sort='stars')
    # query = g.search_repositories('opencl language:c++ stars:>10', sort='stars')
    repos = []

    for repo in query:
        c = conn.cursor()

        print(repo.name, repo.stargazers_count)

        # Record repo in table.
        fork = 1 if repo.fork else 0
        c.execute(
            "INSERT OR IGNORE INTO repos"
            "(url,author,name,stars,fork,updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (repo.url, repo.owner.name, repo.name, repo.stargazers_count, fork,
             repo.updated_at)
        )

        # Iterate over the entire git tree of the repo's default
        # branch (usually 'master'). If a file ends with the .cl
        # extension, check to see if we already have it, else download
        # it.
        #
        try:
            branch = repo.default_branch

            for f in repo.get_git_tree(branch, recursive=True).tree:
                if f.path.endswith('.cl'):
                    # print('  ', f.path)
                    c.execute(
                        "SELECT url FROM files WHERE url=?", (f.url,)
                    )
                    exists = c.fetchone()

                    if exists:
                        print("  in db", f.path)
                        pass
                    else:
                        print("  downloading file", f.path)
                        response = json.loads(requests.get(
                            f.url,
                            headers = {
                                'Authorization': 'token ' + str(github_token)
                            }
                        ).content.decode('utf-8'))
                        code = b64decode(response['content'])

                        c.execute(
                            "INSERT OR IGNORE INTO files"
                            "(url,path,repo_id,source) VALUES "
                            "(?,?,?,?)",
                            (f.url, f.path, repo.url, code)
                        )
        except GithubException as e:
            # If the repository if empty, do nothing.
            pass
        conn.commit()

    conn.close()


if __name__ == '__main__':
    main()
