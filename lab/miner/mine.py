#!/usr/bin/env python3
import clgen
import datetime
import os
import requests
import sqlalchemy as sql
import subprocess
import sys
import threading
import multiprocessing

from configparser import ConfigParser
from progressbar import ProgressBar
from github import Github, GithubException, Repository
from github.GithubException import RateLimitExceededException
from labm8 import crypto, fs, prof
from pathlib import Path
from sqlalchemy.ext.declarative import declarative_base
from tempfile import TemporaryDirectory
from time import sleep
from typing import List

import dsmith

class Octopus(threading.Thread):
    def __init__(self, language: str, nmax: int, clonedir: Path):
        self.i = 0
        self.n = nmax
        self.clonedir = clonedir
        cfg = ConfigParser()
        cfg.read(fs.path("~/.githubrc"))

        github_username = cfg["User"]["Username"]
        github_pw = cfg["User"]["Password"]

        self.g = Github(github_username, github_pw)
        self.query = self.g.search_repositories(f"language:{language} sort:stars fork:false")
        self.num_repos = self.query.totalCount
        self.page_count = 0
        super(Octopus, self).__init__()

    def next_page(self):
        """ fetch page of query results. block on rate limiter """
        while True:
            try:
                page = self.query.get_page(self.page_count)
                self.page_count += 1
                return page
            except RateLimitExceededException:
                sleep(3)

    def run(self):
        page = self.next_page()
        fs.mkdir(self.clonedir)

        while True:
            num_cloned = len(fs.ls(self.clonedir))
            remaining = max(self.n - num_cloned, 0)
            repos = list(page)[:remaining]

            # empty page implies no more results
            if not len(repos):
                break

            pool = multiprocessing.Pool()
            for _ in pool.imap_unordered(
                    CloneWorker(self.clonedir), repos):
                self.i += 1

            # we've gathered enough repos, or have exhausted the repos on github
            if remaining <= 0 or self.i >= self.num_repos:
                break

            page = self.next_page()


class CloneWorker(object):
    def __init__(self, clonedir: Path):
        self.clonedir = clonedir

    def __call__(self, repo):
        clonedir = fs.path(self.clonedir, repo.name)
        if not os.path.exists(clonedir):
            try:
                # shallow clone of repository
                subprocess.check_call(
                    ["timeout", "900", "git", "clone", "--depth", "1", repo.clone_url, clonedir],
                    stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
                # rm empty files
                subprocess.check_call(
                    ["find", clonedir, "-size", "0", "-delete"],
                    stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
            except subprocess.CalledProcessError:
                print(f"warning: failed to clone {repo.name}", file=sys.stderr)


def run_worker(worker: threading.Thread, title):
    """
    run worker thread with a progress bar. Requires worker has `i` and `n`
    properties showing the number of jobs done and in total, respectively.
    """
    print(title, "...")
    bar = ProgressBar(max_value=worker.n)
    worker.start()
    while worker.is_alive():
        bar.update(worker.i)
        worker.join(.25)
    bar.update(worker.n)


if __name__ == "__main__":
    num_repos = int(sys.argv[1])
    language = sys.argv[2]

    clone_dir = fs.path(f"~/dsmith/miner/repos/{language}")
    octopus = Octopus(language, num_repos, clone_dir)
    num_repos = min(num_repos, octopus.num_repos)
    run_worker(octopus, f"scraping {num_repos} of {octopus.num_repos} {language} repos")
