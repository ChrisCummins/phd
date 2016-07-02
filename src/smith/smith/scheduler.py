#!/usr/bin/env python3
#
import json
import locale
import os
import re
import sqlite3
import subprocess
import sys

from argparse import ArgumentParser
from hashlib import sha1
from random import shuffle
from socket import gethostname

import smith
from smith import explore
from smith import preprocess
from smith import train

torch_dir = '~/src/torch-rnn'

def database_path(db_name):
    db_path_base = 'data'
    return os.path.join(db_path_base, str(db_name) + '.db')


def fetch_samples(sample_path):
    with open(sample_path) as infile:
        contents = infile.read()
        samples = re.split(r'=== SAMPLE [0-9]+ ===', contents)
        return [sample.strip() for sample in samples if sample.strip()]


def checksum(s):
    return sha1(s.encode('utf-8')).hexdigest()


def is_host():
    return gethostname() != "whz4"


class task(object):
    def __init__(self, db, db_path, benchmark, seed, oracle,
                 target_num_samples=50000):
        self.db = db
        self.benchmark = benchmark
        self.seed = seed
        self.oracle = oracle
        self.db_path = db_path
        self.db = db
        self.target_num_samples=target_num_samples
        self.preprocessed = False

    def samples_remaining(self):
        c = self.db.cursor()
        c.execute('SELECT Count(*) FROM ContentFiles')
        num_samples = c.fetchone()[0]
        c.close()
        return max(self.target_num_samples - num_samples, 0)

    def complete(self):
        if is_host():
            return self.preprocessed
        else:
            return self.samples_remaining() == 0

    def next_step(self):
        if is_host():
            self.next_host_step()
        else:
            self.next_device_step()

    def next_host_step(self):
        out_path = os.path.join(
            os.path.dirname(self.db_path),
            os.path.splitext(os.path.basename(self.db_path))[0] + '.txt')
        print(out_path)

        preprocess.preprocess_db(self.db_path)
        train.train(self.db_path, out_path, fileid=True)
        explore.explore(self.db_path)
        self.preprocessed = True

    def next_device_step(self):
        os.chdir(os.path.expanduser(torch_dir))
        print("next step:", str(self))
        cmd = "./give-me-kernels '{}' > /tmp/sample.txt".format(self.seed)
        print('\r\033[K  -> seed:'.format(self.seed), end='')
        subprocess.call(cmd, shell=True)

        print('\r\033[K  -> adding samples to database', end='')
        samples = fetch_samples('/tmp/sample.txt')
        ids = [checksum(sample) for sample in samples]

        c = self.db.cursor()
        for id,sample in zip(ids, samples):
            c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
                      (id,sample))
        self.db.commit()
        c.close()
        print('\r\033[K', end='')


    def __repr__(self):
        return ('{} samples from {} seed'
                .format(self.samples_remaining(), self.benchmark))


def run_task(task):
    print(task)


def run_tasks(tasks):
    for task in tasks:
        print(task)

    while len(tasks):
        tasks = [task for task in tasks if not task.complete()]
        shuffle(tasks)  # work balance
        for task in tasks:
            task.next_step()


def connect_to_database(db_path):
    if not os.path.exists(db_path):
        # TODO: run create-dabatase.sql
        return sqlite3.connect(db_path)
    return sqlite3.connect(db_path)


def create_task(job, target):
    data = job['targets'][target]
    benchmark = data['benchmark']
    seed = data['seed']
    db_path = database_path(target)
    oracle_path = data['oracle']

    db = connect_to_database(db_path)
    with open(oracle_path) as infile:
        oracle = infile.read().strip()

    return task(db, db_path, benchmark, seed, oracle)


def run(job_path):
    if not os.path.exists(job_path):
        print("fatal: file", job_path, "not found")
        sys.exit(1)

    with open(job_path) as infile:
        job = json.load(infile)

        tasks = [create_task(job, target) for target in job['targets']]
        run_tasks(tasks)
