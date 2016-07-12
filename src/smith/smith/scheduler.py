import json
import locale
import os
import re
import sqlite3
import subprocess
import sys

from random import shuffle

import smith
from smith import config
from smith import explore
from smith import preprocess
from smith import torch_rnn
from smith import train


class PrototypeException(smith.SmithException): pass


def load_oracle(path):
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        print("fatal: oracle file '", path, "' not found",
              sep='', file=sys.stderr)
        sys.exit(1)

    with open(path) as infile:
        oracle = infile.read().strip()

    return oracle


def extract_prototype(implementation):
    """
    Extract OpenCL kernel prototype from rewritten file.

    :param implementation: OpenCL kernel implementation.
    :return: Kernel prototype.
    """
    if not implementation.startswith("__kernel void A"):
        raise PrototypeException("malformed seed '{}'".format(path))

    try:
        index = implementation.index('{') + 1
        prototype = implementation[:index]
    except ValueError:
        raise PrototypeException("malformed seed '{}'".format(path))

    return prototype


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
        oracle_path = os.path.expanduser(data['path'])
        db = connect_to_database(db_path)
        oracle = load_oracle(oracle_path)
        try:
            seed = data['seed']
        except KeyError:
            seed = extract_prototype(oracle)

        task_args = [db_path, benchmark, seed, oracle]
        if config.is_host():
            return host_task(*task_args)
        else:
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
    def __init__(self, *args, target_num_samples=10000, **kwargs):
        super(device_task, self).__init__(*args, **kwargs)
        self.target_num_samples = target_num_samples

    def complete(self):
        return self._samples_remaining() == 0

    def next_step(self):
        os.chdir(os.path.expanduser(config.torch_rnn_path()))
        print("next step:", str(self))

        num_samples = min(1000, self._samples_remaining())
        samples = torch_rnn.opencl_samples(self.seed, num_samples=num_samples)
        ids = [smith.checksum_str(sample) for sample in samples]

        db = sqlite3.connect(self.db_path)
        c = db.cursor()
        for id,sample in zip(ids, samples):
            c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
                      (id,sample))
        self.db.commit()
        c.close()
        db.close()

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
    """
    Connect to database. If database does not exist, create it.

    :param db_path: Path to database file
    :return: Connection to database.
    """
    if not os.path.exists(db_path):
        db = sqlite3.connect(db_path)
        c = db.cursor()
        script = smith.sql_script('create-samples-db')
        c.executescript(script)
        c.close()
        return db
    else:
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
            return json.loads(smith.minify_json(infile.read()))
        except ValueError as e:
            print("fatal: malformed JSON file '{}'. Error:".format(path),
                  file=sys.stderr)
            print("   ", e, file=sys.stderr)
            sys.exit(1)


def run(job_path):
    job = load_json(job_path)
    data_path = job['data']['path']

    tasks = [task.from_json(task_db_path(data_path, target), job, target)
             for target in job['targets']]
    run_tasks(tasks)
