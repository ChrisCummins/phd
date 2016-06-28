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


class task(object):
    def __init__(self, db, benchmark, seed, oracle, target_num_samples=10000):
        self.db = db
        self.benchmark = benchmark
        self.seed = seed
        self.oracle = oracle
        self.db = db
        self.target_num_samples=target_num_samples

    def samples_remaining(self):
        c = self.db.cursor()
        c.execute('SELECT Count(*) FROM ContentFiles')
        num_samples = c.fetchone()[0]
        c.close()
        return max(self.target_num_samples - num_samples, 0)

    def complete(self):
        return self.samples_remaining() == 0

    def next_step(self):
        print("next step:", str(self))
        subprocess.call("./give-me-kernels '{}' > /tmp/sample.txt"
                        .format(self.seed), shell=True)
        samples = fetch_samples('/tmp/sample.txt')
        ids = [checksum(sample) for sample in samples]

        c = self.db.cursor()
        for id,sample in zip(ids, samples):
            c.execute('INSERT INTO ContentFiles VALUES(?,?)', (id,sample))
        self.db.commit()
        c.close()


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
        os.chdir(os.path.expanduser(torch_dir))
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

    return task(db, benchmark, seed, oracle)


def main():
    locale.setlocale(locale.LC_ALL, 'en_GB.utf-8')

    parser = ArgumentParser()
    parser.add_argument('input', help='path to job description')
    args = parser.parse_args()

    job_path = os.path.expanduser(args.input)

    if not os.path.exists(job_path):
        print("fatal: file", job_path, "not found")
        sys.exit(1)

    with open(job_path) as infile:
        job = json.load(infile)

        tasks = [create_task(job, target) for target in job['targets']]
        run_tasks(tasks)


if __name__ == '__main__':
    main()
