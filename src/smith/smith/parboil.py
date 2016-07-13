"""
Module for Parboil experiments.
"""
import smith

from smith import config
from smith import InternalException

import os
import re
import sqlite3
import subprocess
import sys

from os import remove, close
from shutil import move
from socket import gethostname
from subprocess import CalledProcessError
from tempfile import mkstemp

#
# Exception types
#
class ParboilException(smith.SmithException): pass
class DatabaseException(ParboilException): pass
class BenchmarkException(ParboilException): pass

class OpenCLDeviceType(object):
    GPU = "CL_DEVICE_TYPE_GPU"
    CPU = "CL_DEVICE_TYPE_CPU"

    @staticmethod
    def to_str(devtype):
        if devtype == "CL_DEVICE_TYPE_GPU":
            return "GPU"
        elif devtype == "CL_DEVICE_TYPE_CPU":
            return "CPU"
        else:
            raise InternalException("Invalid OpenCLDeviceType '{}'"
                                    .format(devtype))

class OpenCLDevice(object):
    def __init__(self, name, devtype=OpenCLDeviceType.GPU):
        self.name = name
        self.type = devtype

    def __repr__(self):
        return str(self.name)[:20]


class ScenarioStatus(object):
    GOOD = 0
    BAD = 1
    UNKNOWN = 2

    @staticmethod
    def to_str(status):
        if status == 0:
            return "GOOD"
        elif status == 1:
            return "BAD"
        elif status == 2:
            return "UNKNOWN"
        else:
            raise InternalException("Invalid ScenarioStatus '{}'"
                                    .format(status))


def replace(path, pattern, subst):
    path = os.path.expanduser(path)

    # Create temp file:
    fh, out_path = mkstemp()
    # Write file line by line:
    with open(out_path,'w') as new_file:
        with open(path) as old_file:
            for line in old_file:
                new_file.write(re.sub(pattern, subst, line))
    close(fh)
    # Remove original file:
    remove(path)
    # Move new file:
    move(out_path, path)


class Kernel(object):
    def __init__(self, kernel_id):
        self.id = str(kernel_id)

    def __repr__(self):
        return self.id


class Dataset(object):
    def __init__(self, dataset_id):
        self.id = str(dataset_id)

    def __repr__(self):
        return self.id


class Scenario(object):
    def __init__(self, device_name, benchmark, kernel, dataset,
                 status=ScenarioStatus.UNKNOWN):
        """
        Create Scenario.

        :param device_name: Device name as a string.
        :param benchmark: Benchmark class instance
        :param kernel: Kernel class instance
        :param dataset: Dataset class instance
        :param status (optional): Scenario enum
        """
        self.id = Scenario.get_id(device_name, benchmark, kernel, dataset)
        self.device = device_name
        self.benchmark = benchmark
        self.kernel = kernel
        self.dataset = dataset
        self.status = status

    def __repr__(self):
        return ("{}:{}:{}: {}-{}"
                .format(self.benchmark, self.kernel, self.dataset,
                        OpenCLDeviceType.to_str(self.device),
                        ScenarioStatus.to_str(self.status)))

    @staticmethod
    def get_id(device, benchmark, kernel, dataset):
        return smith.checksum_str(repr(device) + repr(benchmark) +
                                  repr(kernel) + repr(dataset))


class Benchmark(object):
    def __init__(self, benchmark_id):
        parboil_root = config.parboil_root()

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

        src_file = ImplementationFile.from_benchmark(parboil_root, benchmark_id)

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

    def run(self, kernel, dataset, device=OpenCLDeviceType.GPU, n=30):
        """
        Run a benchmark for a number of iterations, returning the runtimes.

        :param kernel: Kernel class instance
        :param dataset: Dataset class instance
        :param device: OpenCLDeviceType enum
        :param n: Number of iterations to run for
        :return: List of 'n' Runtime class instances
        :throws: BenchmarkException in case of error during
                 compilation or execution
        """
        if dataset.id not in [x.id for x in self.datasets]:
            raise BenchmarkException("No such dataset '{}'".format(dataset))

        # Set the execution device
        self.src_file.set_device_type(device)

        # Set the kernel
        # TODO: self.set_kernel(kernel.contents())

        # Build the benchmark
        os.chdir(self.parboil_root)
        print("building", self.id, self.implementation, "...")
        cmd = ("./parboil compile {} {} >/dev/null"
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
                print("runtime", runtime)
                runtimes.append(runtime)
            except CalledProcessError as e:
                print(e, file=sys.stderr)
                raise BenchmarkException("Benchmark execution failed")
        return runtimes


    def __repr__(self):
        return str(self.id)


class Runtime(object):
    """
    Runtime result.
    """
    def __init__(self, scenario, io, kernel, copy, driver, compute, overlap,
                       wall):
        """
        Create a new Runtime instance.

        :param scenario: Scenario class instance.
        :param io: IO time, as reported by Parboil.
        :param kernel: Kernel execution time, as reported by Parboil.
        :param copy: Copy time, as reported by Parboil.
        :param driver: Driver time, as reported by Parboil.
        :param compute: Compute ratio, as reported by Parboil.
        :param overlap: CPU/GPU overlap, as reported by Parboil.
        :param wall: Wall clock runtime, as reported by Parboil.
        """
        self.scenario = scenario
        self.io = io
        self.kernel = kernel
        self.copy = copy
        self.driver = driver
        self.compute = compute
        self.overlap = overlap
        self.wall = wall

    def __repr__(self):
        return repr(self.scenario)

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

    @staticmethod
    def from_stdout(benchmark, kernel, dataset, device, stdout):
        """
        Create a new Runtime instance.

        :param benchmark: Benchmark class instance.
        :param kernel: Kernel class instance.
        :param dataset: Dataset class instance.
        :param device: OpenCLDeviceType class instance.
        :param stdout: str of Parboil benchmark output.
        :return: Runtime instance.
        """
        scenario = Scenario(device, benchmark, kernel, dataset,
                            ScenarioStatus.GOOD)

        # TODO: extract from stdout
        io = 2
        kernel = 2
        copy = 2
        driver = 2
        compute = 2
        overlap = 2
        wall = 2

        return Runtime(scenario, io, kernel, copy, driver, compute, overlap,
                       wall)


class Database(object):
    """
    Database abstraction for Parboil experiment results.
    """
    VERSION = 1

    def __init__(self, db_path):
        db_path = os.path.abspath(os.path.expanduser(db_path))
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
        return [Benchmark(x) for x in benchmark_ids]

    def add_kernel(self, benchmark, contents, oracle=False):
        """
        Add a kernel to the database.

        :param benchmark: Benchmark class instance.
        :param contents: OpenCL source code
        :param oracle: True if oracle implementation, else false
        """
        contents = contents.strip()
        kernel_id = smith.checksum_str(contents)
        benchmark_id = benchmark.id
        oracle = 1 if oracle else 0

        db = self.db()
        c = db.cursor()
        c.execute("INSERT OR IGNORE INTO Kernels VALUES(?,?,?,?)",
                  (kernel_id, benchmark_id, oracle, contents))
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

    def kernels(self, benchmark):
        """
        Return the kernels which match the given status.

        :param benchmark: Benchmark class instance.
        """
        c = self.cursor()
        c.execute('SELECT id FROM Kernels WHERE benchmark=?', (benchmark.id,))
        kernel_ids = [x[0] for x in c.fetchall()]
        kernels = [Kernel(x) for x in kernel_ids]
        c.close()
        return kernels


    def add_runtime(self, runtime):
        """
        Record a new runtime.

        :param runtime: Runtime class instance.
        """
        db = self.db()
        c = db.cursor()
        scenario = runtime.scenario
        host = gethostname()
        c.execute("INSERT OR IGNORE INTO Scenarios VALUES(?,?,?,?,?,?,?)",
                  (scenario.id, host, scenario.device,
                   scenario.benchmark.id, scenario.kernel.id,
                   scenario.dataset.id, scenario.status))

        c.execute("INSERT INTO Runtimes VALUES(?,?,?,?,?,?,?,?)",
                  (scenario.id,
                   runtime.io, runtime.kernel, runtime.copy, runtime.driver,
                   runtime.compute, runtime.overlap, runtime.wall))
        c.close()
        db.commit()

    def add_scenario(self, status=ScenarioStatus.UNKNOWN):
        """
        :param status: ScenarioStatus enum
        """
        pass

    def add_runtimes(self, runtimes):
        """
        Record new runtimes.

        :param runtimes: Iterable sequence of Runtime class instances.
        """
        for runtime in runtimes:
            self.add_runtime(runtime)

    def mark_kernel_bad(self, kernel):
        print("Naughty kernel", kernel.id)

    @staticmethod
    def create_new(path):
        """
        Create a new Parboil database.

        :param path: Path to database to create.
        :return: Parboil class instance of newly created database.
        :throws DatabaseException: If file already exists.
        """
        path = os.path.expanduser(path)

        if os.path.exists(path):
            raise DatabaseException("Database '{}' already exists"
                                    .format(path))

        print("creating database ...".format(path))
        db = sqlite3.connect(path)
        c = db.cursor()
        script = smith.sql_script('create-parboil-db')
        c.executescript(script)
        c.close()
        return Database(path)

    @staticmethod
    def init(path):
        """
        Load an existing Parboil database, or create a new one.

        :param path: Path to database.
        :return: Parboil class instance.
        :throws DatabaseException: In case of error.
        """
        path = os.path.expanduser(path)

        if os.path.exists(path):
            return Database(path)
        else:
            return Database.create_new(path)


class ImplementationFile(object):
    def __init__(self, path):
        path = os.path.expanduser(path)

        # Check that file exists
        if not os.path.exists(path):
            raise Exception("Parboil implementation file '{}' not found"
                            .format(path))

        # Set member variables
        self.path = path

    def set_device_type(self, devtype):
        """
        Modify the underlying implementation to use the given device type.

        :param devtype: OpenCLDeviceType enum
        """
        replace(self.path,
                re.compile('#define MY_DEVICE_TYPE .+'),
                '#define MY_DEVICE_TYPE ' + str(devtype))

    @staticmethod
    def from_benchmark(parboil_root, benchmark_id):
        src_dir = os.path.join(parboil_root, 'benchmarks', benchmark_id, 'src',
                               'opencl_base')
        c_file = os.path.join(src_dir, 'main.c')
        if os.path.exists(c_file):
            return ImplementationFile(c_file)

        cpp_file = os.path.join(src_dir, 'main.cpp')
        if os.path.exists(cpp_file):
            return ImplementationFile(cpp_file)
