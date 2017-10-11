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


# Global state to manage database connections. Must call init() before
# creating sessions.
Base = declarative_base()
engine = None
make_session = None


session_t = sql.orm.session.Session


class Repository(Base):
    id_t = sql.Integer

    __tablename__ = "repositories"
    id = sql.Column(id_t, primary_key=True)
    url = sql.Column(sql.String(256), nullable=False, unique=True, index=True)
    owner = sql.Column(sql.String(256), nullable=False)
    name = sql.Column(sql.String(256), nullable=False)
    fork = sql.Column(sql.Boolean, nullable=False)
    stars = sql.Column(sql.Integer, nullable=False)
    num_forks = sql.Column(sql.Integer, nullable=False)
    created_at = sql.Column(sql.DateTime, nullable=False)
    updatd_at = sql.Column(sql.DateTime, nullable=False)


class ContentFile(Base):
    id_t = sql.Integer

    __tablename__ = "content_files"
    id = sql.Column(id_t, primary_key=True)
    repo = sql.Column(Repository.id_t, sql.ForeignKey("repositories.id"), nullable=False)
    sha1 = sql.Column(sql.String(40), nullable=False, unique=True, index=True)
    date_added = sql.Column(sql.DateTime, nullable=False, default=datetime.datetime.utcnow)
    linecount = sql.Column(sql.Integer, nullable=False)
    charcount = sql.Column(sql.Integer, nullable=False)
    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)


class PreprocessedFile(Base):
    id_t = sql.Integer

    __tablename__ = "preprocessed_files"
    id = sql.Column(id_t, sql.ForeignKey("content_files.id"), primary_key=True)
    sha1 = sql.Column(sql.String(40), nullable=False, unique=True, index=True)
    date_added = sql.Column(sql.DateTime, nullable=False, default=datetime.datetime.utcnow)
    linecount = sql.Column(sql.Integer, nullable=False)
    charcount = sql.Column(sql.Integer, nullable=False)
    src = sql.Column(sql.UnicodeText(length=2**31), nullable=False)
    returncode = sql.Column(sql.Integer, nullable=False)
    stderr = sql.Column(sql.UnicodeText(length=2**31), nullable=False)


def init(hostname: str, echo: bool=False) -> str:
    """
    Initialize database engine.

    Must be called before attempt to create a database connection.

    Arguments:
        hostname (str): Hostname of machine running MySQL database.

    Returns:
        str: URI of database.
    """
    global engine
    global make_session
    username, password = dsmith.MYSQL_CREDENTIALS
    schema = "GitHub_C"
    port = str(dsmith.PORT)

    # Use UTF-8 encoding (default is latin-1) when connecting to MySQL.
    # See: https://stackoverflow.com/a/16404147/1318051
    uri = f"mysql+mysqldb://{username}:{password}@{hostname}:{port}/{schema}?charset=utf8"
    echo = True if echo else True if os.environ.get("ECHO") else False
    engine = sql.create_engine(uri, encoding="utf-8", echo=echo)

    Base.metadata.create_all(engine)
    Base.metadata.bind = engine
    make_session = sql.orm.sessionmaker(bind=engine)

    profile = True if os.environ.get("PROF") else False
    if profile:
        prof.enable()

    return f"mysql://{hostname}:{port}/{schema}"


class Octopus(threading.Thread):
    def __init__(self, language: str, nmax: int):
        self.i = 0
        self.n = nmax
        self.repos = []
        cfg = ConfigParser()
        cfg.read(fs.path("~/.githubrc"))

        github_username = cfg["User"]["Username"]
        github_pw = cfg["User"]["Password"]

        self.g = Github(github_username, github_pw)
        self.query = self.g.search_repositories(f"language:{language} fork:false sort:stars")
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

        while True:
            remaining = self.n - self.i
            self.repos += list(page)[:remaining]
            self.i = len(self.repos)

            if len(self.repos) >= self.n:
                break

            page = self.next_page()


class ScraperWorker(object):
    def __init__(self, clonedir: Path, destdir: Path, extensions: List[str],
                 nmax: int):
        self.clonedir = clonedir
        self.destdir = destdir
        self.extensions = extensions
        self.nmax = nmax

    def __call__(self, repo):
        clonedir = fs.path(self.clonedir, repo.name)

        if not os.path.exists(clonedir):
            subprocess.check_output(
                ["git", "clone", "--depth", "1", repo.clone_url, clonedir],
                stderr=subprocess.STDOUT)

        names = " -o ".join([f"-name '*{ext}'" for ext in self.extensions])
        out = subprocess.check_output(
            f"find {clonedir} -type f {names} | " +
            f"head -n {self.nmax}", universal_newlines=True, shell=True)

        for path in out.split("\n"):
            if path:
                extension = path.split(".")[-1]
                sha1 = crypto.sha1_file(path)
                dest = f"{self.destdir}/{sha1}.{extension}"
                fs.cp(path, dest)


class Scraper(threading.Thread):
    def __init__(self, repos: List[Repository], clonedir: Path, outdir: Path,
                 extensions: List[str], max_files_per_repo: int):
        self.clonedir = clonedir
        self.repos = repos
        self.outdir = outdir
        self.i = 0
        self.n = len(repos)
        self.extensions = extensions
        self.max_files_per_repo = max_files_per_repo
        fs.mkdir(self.outdir)
        fs.mkdir(self.clonedir)
        super(Scraper, self).__init__()

    def run(self):
        pool = multiprocessing.Pool()
        for i, _ in enumerate(pool.imap_unordered(
                ScraperWorker(self.clonedir, self.outdir, self.extensions,
                              self.max_files_per_repo),
                self.repos)):
            self.i = i
        subprocess.check_output(["find", self.outdir, "-size", "0", "-delete"])


class PreprocessorWorker(object):
    def __init__(self, destdir: Path, timeout: int=60):
        self.destdir = destdir
        self.timeout = timeout

    def __call__(self, path: Path):
        dest = fs.path(self.destdir, fs.basename(path))
        if not os.path.exists(dest):
            cmd = f"""
timeout -s9 {self.timeout} \
{clgen.native.CLGEN_REWRITER} {path} 2>/dev/null | \
gcc -fpreprocessed -dD -E - 2>/dev/null | \
{clgen.native.CLANG_FORMAT} 2>/dev/null > {dest}
"""
            subprocess.call(cmd, shell=True)


class Preprocessor(threading.Thread):
    def __init__(self, indir: Path, outdir: Path):
        self.indir = indir
        self.outdir = outdir
        self.i = 0
        self.paths = fs.ls(self.indir, abspaths=True)
        self.n = len(self.paths)
        fs.mkdir(self.outdir)
        super(Preprocessor, self).__init__()

    def run(self):
        pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
        for i, _ in enumerate(pool.imap_unordered(PreprocessorWorker(self.outdir), self.paths)):
            self.i = i
        subprocess.check_output(["find", self.outdir, "-size", "0", "-delete"])


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
    init(dsmith.MYSQL_HOSTNAME)

    clone_dir = "miner/repos"
    file_extensions = [".c"]
    files_per_repo = 1000
    language = "c"
    num_repos = 1000
    preprocessed_dir = "miner/preprocessed"
    sources_dir = "miner/sources"

    spider = Octopus(language, num_repos)
    run_worker(spider, "finding repos to scrape")
    run_worker(Scraper(spider.repos, clone_dir, sources_dir, file_extensions,
                       files_per_repo),
               "scraping repos")
    run_worker(Preprocessor(sources_dir, preprocessed_dir),
               "preprocessing files")
