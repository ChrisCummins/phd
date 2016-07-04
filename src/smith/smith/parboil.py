"""
Module for Parboil experiments.
"""

import os
import re
import sqlite3
import subprocess
import sys

from hashlib import sha1
from os import remove, close
from shutil import move
from subprocess import CalledProcessError
from tempfile import mkstemp

# Number of Platforms found: 1
#   Number of Devices found for Platform 0: 1
# Chose Device Type: GPU
# Starting GPU kernel
# GPU kernel done
# IO        : 0.870675
# Kernel    : 0.174193
# Copy      : 0.724049
# Driver    : 0.174197
# CPU/Kernel Overlap: 0.174211
# Timer Wall Time: 1.768981
# Pass

#
# Exception types
#
class ParboilException(Exception): pass
class DatabaseException(ParboilException): pass
class BenchmarkException(ParboilException): pass

class Device(object):
    GPU = "CL_DEVICE_TYPE_GPU"
    CPU = "CL_DEVICE_TYPE_CPU"

class KernelStatus(object):
    GOOD = 0
    BAD = 1
    UNKNOWN = 2


def checksum(s):
    return sha1(s.encode('utf-8')).hexdigest()


def checksum_file(path):
    with open(path) as infile:
        return checksum(infile.read())


def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(re.sub(pattern, subst, line))
    close(fh)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)


class Kernel(object):
    def __init__(self, kernel_id):
        self.id = kernel_id

    def __repr__(self):
        return str(self.id)


class Dataset(object):
    def __init__(self, dataset_id):
        self.id = dataset_id

    def __repr__(self):
        return str(self.id)


class Scenario(object):
    def __init__(self, device, benchmark, kernel, dataset):
        self.id = Scenario.get_id(device, benchmark, kernel, dataset)
        self.device = device
        self.benchmark = benchmark
        self.kernel = kernel
        self.dataset = dataset

    def __repr__(self):
        return str(self.id)

    @staticmethod
    def get_id(device, benchmark, kernel, dataset):
        return hash(repr(device) + repr(benchmark) + repr(kernel) +
                    repr(dataset))


class Benchmark(object):
    def __init__(self, parboil_root, benchmark_id):
        # Check that parboil exists
        parboil_root = os.path.expanduser(parboil_root)
        if not os.path.exists(parboil_root):
            raise BenchmarkException("Parboil root '{}' not found"
                                     .format(parboil_root))

        # Check that benchmark exists
        if not os.path.exists(os.path.join(parboil_root, 'benchmarks',
                                           benchmark_id)):
            raise BenchmarkException("Parboil benchmark '{}' not found"
                                     .format(benchmark_id))

        # Check that opencl_base or equivalent exists
        implementation = 'opencl_base'
        if not os.path.exists(os.path.join(
                parboil_root, 'benchmarks', benchmark_id, 'src',
                implementation)):
            raise BenchmarkException("Parboil implementation '{}' not found"
                                     .format(implementation))

        src_file = ImplementationFile(
            os.path.join(parboil_root, 'benchmarks', benchmark_id,
                         'src', implementation, 'main.c'))

        # Get datasets
        datasets_path = os.path.join(parboil_root, 'datasets', benchmark_id)
        datasets = [Dataset(x) for x in os.listdir(datasets_path)]

        # Get OpenCL kernel
        oracle_kernel_path = os.path.join(parboil_root, 'benchmarks',
                                          benchmark_id, 'src', implementation,
                                          'kernel.cl')
        if not os.path.exists(oracle_kernel_path):
            raise BenchmarkException("Parboil OpenCL kernel '{}' not found"
                                     .format(oracle_kernel_path))

        # Set member variables
        self.parboil_root = parboil_root
        self.id = benchmark_id
        self.implementation = implementation
        self.datasets = datasets
        self.oracle_kernel_path = oracle_kernel_path
        self.src_file = src_file

    def __repr__(self):
        return str(self.id)

    def run(self, kernel, dataset, device=Device.GPU, n=30):
        """
        Run a benchmark for a number of iterations, returning the runtimes.

        :param kernel: Kernel class instance
        :param dataset: Dataset class instance
        :param device: Device enum
        :param n: Number of iterations to run for
        :return: List of 'n' Runtime class instances
        :throws: BenchmarkException in case of error during
                 compilation or execution
        """
        if dataset not in self.datasets:
            raise BenchmarkException("No such dataset '{}'".format(dataset))

        # Set the execution device
        self.src_file.set_device_type(device)

        # Set the kernel
        # TODO: self.set_kernel(kernel.contents())

        # Build the benchmark
        os.chdir(self.parboil_root)
        cmd = ("./parboil compile {} {}"
               .format(self.id, self.implementation))
        ret = subprocess.call(cmd, shell=True)
        if ret:
            raise BenchmarkException("Benchmark compilation failed")

        runtimes = []
        for i in range(n):
            cmd = ("./parboil run {} {} {}"
                   .format(self.id, self.implementation, dataset))
            try:
                out = subprocess.check_output(cmd, shell=True).decode()
                runtime = Runtime.from_stdout(
                    self, kernel, dataset, device, out)
                results.append(result)
            except CalledProcessError as e:
                print(e, file=sys.stderr)
                raise BenchmarkException("Benchmark execution failed")
        return runtimes


    def __repr__(self):
        return str(self.id)


class Runtime(object):
    def __init__(self, scenario, io, kernel, copy, driver, copmute, overlap,
                       wall):
        self.scenario
        self.io
        self.kernel
        self.copy
        self.driver
        self.copmute
        self.overlap
        self.wall

    def __repr__(self):
        return "{} {} {}".format(self.scenario)

    @staticmethod
    def from_stdout(benchmark, kernel, dataset, device, stdout):
        scenario = Scenario(device, benchmark, kernel, dataset)

        # TODO: extract from stdout
        io = 2
        kernel = 2
        copy = 2
        driver = 2
        compute = 2
        overlap = 2
        wall = 2

        return Runtime(scenario, io, kernel, copy, driver, copmute, overlap,
                       wall)


class Database(object):
    VERSION = 1

    def __init__(self, db_path):
        db_path = os.path.expanduser(db_path)
        if not os.path.exists(db_path):
            raise DatabaseException("Database '{}' not found".format(db_path))

        print("loading database '{}' ...".format(db_path))
        try:
            db = sqlite3.connect(db_path)
            c = db.cursor()
            c.execute('SELECT value FROM Meta WHERE key=?', ("version",))
            version = int(c.fetchone()[0])
        except Exception as e:
            raise DatabaseException("Malformed database '{}'".format(db_path))

        if version != self.VERSION:
            raise DatabaseException("Database version {}, expected {}"
                                    .format(version, self.VERSION))

        c.execute('SELECT value FROM Meta WHERE key=?', ('parboil-root',))
        parboil_root = c.fetchone()[0]

        self.db_path = db_path
        self.parboil_root = parboil_root

        # Initialisation
        self._add_oracle_kernels()

    def db(self):
        """
        Get a connection to the database.
        """
        return sqlite3.connect(self.db_path)

    def cursor(self):
        """
        Create a database connection and return cursor.

        Note that if you want to modify the contents of the database,
        you must instead call self.db() and then commit the
        changes. This method provides read access only.
        """
        return self.db().cursor()

    def benchmarks(self):
        """
        Return the benchmark objects.
        """
        c = self.cursor()
        c.execute("SELECT id FROM Benchmarks")
        benchmark_ids = [x[0] for x in c.fetchall()]
        c.close()
        return [Benchmark(self.parboil_root, x) for x in benchmark_ids]

    def add_kernel(self, benchmark, contents, oracle=False,
                   status=KernelStatus.UNKNOWN):
        """
        Add a kernel to the database.

        :param benchmark: Benchmark class instance.
        :param contents: OpenCL source code
        :param oracle: True if oracle implementation, else false
        :param status: KernelStatus enum
        """
        contents = contents.strip()
        kernel_id = checksum(contents)
        benchmark_id = benchmark.id
        oracle = 1 if oracle else 0
        if status > 2 or status < 0:
            raise DatabaseException("Invalid kernel status '{}'".format(status))

        db = self.db()
        c = db.cursor()
        c.execute("INSERT OR IGNORE INTO Kernels VALUES(?,?,?,?,?)",
                  (kernel_id, benchmark_id, oracle, contents, status))
        db.commit()

    def _add_oracle_kernels(self):
        """
        Add oracle kernels to database.
        """
        for benchmark in self.benchmarks():
            oracle_path = benchmark.oracle_kernel_path
            with open(oracle_path) as infile:
                contents = infile.read().strip()
                self.add_kernel(benchmark, contents, oracle=True)

    def kernels(self, benchmark, status=0):
        """
        Return the kernels which match the given status.

        :param benchmark: Benchmark class instance.
        """
        c = self.cursor()
        c.execute('SELECT id FROM Kernels WHERE benchmark=? AND status=?',
                  (benchmark.id, status))
        kernel_ids = [x[0] for x in c.fetchall()]
        kernels = [Kernel(x) for x in kernel_ids]
        c.close()
        return kernels


    def add_runtime(self, runtime):
        """
        Record a new runtime.

        :param runtime: Runtime class instance.
        """
        scenario = runtime.scenario
        c.execute("INSERT OR IGNORE INTO Scenarios VALUES(?,?,?,?,?,?)",
                  (scenario.id, host, device, benchmark, kernel))

        c.execute("INSERT INTO Runtimes VALUES(?,?,?,?,?,?,?,?)",
                  (scenario_id,
                   io, kernel, copy, driver, compute, overlap, wall))

    def add_runtimes(self, runtimes):
        for runtime in runtimes:
            self.add_runtime(runtime)

    def mark_kernel_bad(self, kernel_id):
        print("Naughty kernel", kernel_id)


class ImplementationFile(object):
    def __init__(self, path):
        path = os.path.expanduser(path)

        # Check that file exists
        if not os.path.exists(path):
            raise KernelException("Parboil implementation file '{}' not found"
                                  .format(path))

        # Set member variables
        self.path = path

    def set_device_type(self, devtype):
        replace(self.path,
                re.compile('#define MY_DEVICE_TYPE .+'),
                '#define MY_DEVICE_TYPE ' + str(devtype))
