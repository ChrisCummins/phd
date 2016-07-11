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

import smith
from smith import explore
from smith import preprocess
from smith import train
from smith import config


def fetch_samples(sample_path):
    with open(sample_path) as infile:
        contents = infile.read()
        samples = re.split(r'=== SAMPLE [0-9]+ ===', contents)
        return [sample.strip() for sample in samples if sample.strip()]


def checksum(s):
    return sha1(s.encode('utf-8')).hexdigest()


def load_oracle(path):
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        print("fatal: oracle file '", path, "' not found",
              sep='', file=sys.stderr)
        sys.exit(1)

    with open(path) as infile:
        oracle = infile.read().strip()


class task(object):
    def __init__(self, db_path, benchmark, seed, oracle):
        self.benchmark = benchmark
        self.seed = seed
        self.oracle = oracle
        self.db_path = db_path

    def complete(self):
        raise NotImplementedError("Abstract class")

    def next_step(self):
        raise NotImplementedError("Abstract class")

    def __repr__(self):
        return "task"

    @staticmethod
    def from_json(db_path, job, target):
        """
        Task factory.
        """
        data = job['targets'][target]
        benchmark = data['benchmark']
        seed = data['seed']
        oracle_path = data['oracle']

        db = connect_to_database(db_path)
        oracle = load_oracle(oracle_path)

        task_args = [db_path, benchmark, seed, oracle]
        if config.is_host():
            return host_task(*task_args)
        else:
            task_args = [job['torch-rnn']['path']] + task_args
            return device_task(*task_args)


class host_task(task):
    def __init__(self, *args, **kwargs):
        super(host_task, self).__init__(*args, **kwargs)
        self.preprocessed = False

    def complete(self):
        return self.preprocessed

    def next_step(self):
        out_path = os.path.join(
            os.path.dirname(self.db_path),
            os.path.splitext(os.path.basename(self.db_path))[0] + '.txt')
        print(out_path)

        preprocess.preprocess_db(self.db_path)
        train.train(self.db_path, out_path, fileid=True)
        explore.explore(self.db_path)
        self.preprocessed = True


class device_task(task):
    def __init__(self, torch_rnn_path, *args,
                 target_num_samples=1000000, **kwargs):
        super(device_task, self).__init__(*args, **kwargs)
        self.torch_rnn_path = torch_rnn_path
        self.target_num_samples = target_num_samples

    def complete(self):
        return self._samples_remaining() == 0

    def next_step(self):
        os.chdir(os.path.expanduser(self.torch_rnn_path))
        print("next step:", str(self))
        # TODO: Invoke torch-rnn wrapper
        cmd = 'th sample.lua -checkpoint $(ls -t cv/*.t7 | head -n1) -temperature .75 -length 5000 -opencl 1 -start_text "$1" -n 1000'
        print('\r\033[K  -> seed:'.format(self.seed), end='')
        subprocess.call(cmd, shell=True)

        print('\r\033[K  -> adding samples to database', end='')
        samples = fetch_samples('/tmp/sample.txt')
        ids = [checksum(sample) for sample in samples]

        db = sqlite3.connect(self.db_path)
        c = db.cursor()
        for id,sample in zip(ids, samples):
            c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
                      (id,sample))
        self.db.commit()
        c.close()
        db.close()
        print('\r\033[K', end='')

    def _samples_remaining(self):
        db = sqlite3.connect(self.db_path)
        c = db.cursor()
        c.execute('SELECT Count(*) FROM ContentFiles')
        num_samples = c.fetchone()[0]
        c.close()
        db.close()
        return max(self.target_num_samples - num_samples, 0)

    def __repr__(self):
        return ('{} samples from {} seed'
                .format(self._samples_remaining(), self.benchmark))


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


def task_db_path(root, task_name):
    return os.path.expanduser(os.path.join(root, str(task_name) + '.db'))


def load_json(path):
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        print("fatal: JSON file '", path, "' not found",
              sep='', file=sys.stderr)
        sys.exit(1)

    with open(path) as infile:
        try:
            return json.loads(smith.json_minify(infile.read()))
        except ValueError as e:
            print("fatal: malformed JSON file '", path, "'. Error:",
                  sep='', file=sys.stderr)
            print("   ", e, file=sys.stderr)
            sys.exit(1)


def run(job_path):
    job = load_json(job_path)
    data_path = job['data']['path']

    tasks = [task.from_json(task_db_path(data_path, target), job, target)
             for target in job['targets']]
    run_tasks(tasks)
