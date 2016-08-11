#
# libcecl utilities.
#
# WARNING: This code is custom tailored to one specific experimental
# setup and methodology. It is EXTREMELY fragile code!
#
from __future__ import division,absolute_import,print_function,unicode_literals

import os
import re
import six
import sys

from collections import defaultdict
from itertools import product
from multiprocessing import cpu_count, Pool
from random import shuffle

import labm8
from labm8 import fs
from labm8 import math as labmath

import smith
from smith import clutil


class CeclException(smith.SmithException): pass
class LogException(CeclException): pass
class NameInferenceException(CeclException): pass


def assert_device_type(expected, actual):
    if expected.lower() != actual.lower():
        raise LogException("expected device type '{}', found device type '{}'"
                           .format(expected, actual))


class KernelInvocation(object):
    def __init__(self, name, global_size, local_size, runtime,
                 transfer=None, dataset=None):
        self.name = name
        self.dataset = dataset
        self.transfer = transfer
        self.global_size = global_size
        self.local_size = local_size
        self.runtime = runtime


def kernel_invocations_from_cecl_log(log, devtype=None):
    """
    Interpret and parse the output of a libcecl instrumented application.

    Return: list of Kernel Invocation objects.
    """
    kernel_invocations = []
    transferred_bytes = 0
    transfer_time = 0
    function_prefix, dataset = path_to_benchmark_and_dataset(log)
    function_prefix += "-"

    with open(log) as infile:
        contents = infile.read()

    # Iterate over each line in the log.
    for line in contents.split('\n'):
        # Split line based on ; delimiter into opcode and operands.
        line = line.strip()
        components = [x.strip() for x in line.split(';')]
        opcode, operands = components[0], components[1:]

        # Skip empty lines:
        if not opcode:
            continue

        if devtype and opcode == "clCreateCommandQueue":
            assert_device_type(devtype, operands[0])
        elif opcode == "clEnqueueNDRangeKernel":
            function_name, global_size, local_size, elapsed = operands
            global_size = int(global_size)
            local_size = int(local_size)
            elapsed = float(elapsed)
            ki = KernelInvocation(function_prefix + function_name, global_size,
                                  local_size, elapsed, dataset=dataset)
            kernel_invocations.append(ki)
        elif opcode == "clEnqueueTask":
            function_name, elapsed = operands
            elapsed = float(elapsed)
            ki = KernelInvocation(function_prefix + function_name,
                                  1, 1, elapsed, dataset=dataset)
            kernel_invocations.append(ki)
        elif opcode == "clCreateBuffer":
            size, host_ptr, flags = operands
            size = int(size)
            flags = flags.split("|")
            if "CL_MEM_COPY_HOST_PTR" in flags:
                if "CL_MEM_READ_ONLY" in flags:
                    # host -> device
                    transferred_bytes += size
                else:
                    # device <-> host
                    transferred_bytes += size * 2
            else:
                # device -> host
                transferred_bytes += size
        elif (opcode == "clEnqueueReadBuffer" or
              opcode == "clEnqueueWriteBuffer" or
              opcode == "clEnqueueMapBuffer"):
            destination, size, elapsed = operands
            elapsed = float(elapsed)
            transfer_time += elapsed
        else:
            # Not a line that we're interested in.
            pass

    # Before returning list
    for ki in kernel_invocations:
        ki.transfer = transferred_bytes
        ki.runtime += transfer_time

    return kernel_invocations


class Kernel(object):
    def __init__(self, source):
        self._source = source
        self._args = []
        self._transfers = []
        self._invocations = []

    def __repr__(self):
        return repr(self._source)

    def __str__(self):
        return str(self._source)


def path_to_benchmark_and_dataset(path):
    basename = fs.basename(path)
    if basename.startswith("npb-"):
        components = basename.split('.')
        return (".".join(components[:-1]), components[-1])
    elif basename.startswith("nvidia-"):
        return (
            re.sub(r"(nvidia-4\.2-)ocl([a-zA-Z]+)", r"\1\2", basename),
            "default")
    elif basename.startswith("parboil-"):
        components = basename.split("-")
        return ("-".join(components[:-1]), components[-1])
    else:
        return basename, "default"


def process_cecl_log(log, devtype=None):
    benchmark, dataset = path_to_benchmark_and_dataset(log)
    kernel_invocations = parse_cecl_log(log, devtype=devtype)

    for ki in kernel_invocations:
        name = benchmark + "-" + ki.name
        print(
            name,
            dataset,
            ki.transfer,
            ki.global_size,
            ki.local_size,
            ki.runtime,
            sep=",")


def log2features(log, out=sys.stdout, metaout=sys.stderr):
    process_cecl_log(log)


def get_device_logs(logdir):
    """
    Walks a directory tree of this structure:

    <logdir>/<run>/<device>/<log>

    and returns a dictionary mapping devices to lists of logs:

    { <device>: [<logs ...>] }
    """
    devlogs = defaultdict(list)
    rundirs = [x for x in fs.ls(logdir, abspaths=True) if fs.isdir(x)]
    devices = [fs.basename(x) for x in fs.ls(rundirs[0])]

    for rundir,device in product(rundirs, devices):
        devlogs[device] += sorted(
            [x for x in fs.ls(fs.path(rundir, device), abspaths=True)
             if fs.isfile(x)])

    # The number of files in the logdir:
    nfiles = (len([x for x in fs.ls(logdir, abspaths=True, recursive=True)
                   if fs.isfile(x)]) -
              len([x for x in fs.ls(logdir, abspaths=True) if fs.isfile(x)]))
    # The number of logs found:
    nlogs = sum([len(x) for x in devlogs.values()])

    if nlogs != nfiles:
        raise LogException("There are {} files in the log directory, "
                           "but we found {}. Is the layout correct? ({})"
                           .format(nfiles, nlogs, fs.path(
                               logdir, "<run>", "<device>", "<logs>")))
    if nlogs < 1:
        raise LogException("No logs found! Is the layout correct? ({})"
                           .format(fs.path(
                               logdir, "<run>", "<device>", "<logs>")))

    # Print directory:
    print("libcecl logs:")
    print("    # files:", nfiles)
    print("    # runs:", len(rundirs), "({})".format(
        ", ".join(["{}: {}".format(fs.basename(x),
                                   len([y for y in fs.ls(x, abspaths=True,
                                                         recursive=True)
                                        if fs.isfile(y)]))
                   for x in fs.ls(logdir, abspaths=True)])))
    print("    # devices:", len(devices), "({})".format(
        ", ".join(["{}: {}".format(x, len(devlogs[x])) for x in devlogs])))


    # Find the common intersection of logs which exist for all
    # devices:
    common = set([fs.basename(x) for x in devlogs[devices[0]]])
    for device in devices[1:]:
        common = common.intersection([fs.basename(x) for x in devlogs[device]])
    # Update devlogs to include only common benchmarks:
    for device in devices:
        devlogs[device] = [x for x in devlogs[device]
                           if fs.basename(x) in common]
    nlogs = sum([len(x) for x in devlogs.values()])

    print("common subset:")
    print("    # files:", nlogs, "({:.1f}%)".format((nlogs / nfiles) * 100))
    print("    # common:", len(common))
    print("    # devices:", len(devices), "({})".format(
        ", ".join(["{}: {}".format(x, len(devlogs[x])) for x in devlogs])))
    print()

    return devlogs


def device_to_device_type(devname):
    d = {
        "amd": "GPU",
        "intel": "CPU",
        "nvidia": "GPU"
    }.get(devname, None)
    if d is None:
        raise LogException("Unknown device name '{}'"
                           .format(devname))
    return d


def _log_worker(job):
    return kernel_invocations_from_cecl_log(
        job["path"], devtype=job["devtype"])


def _log_reducer(job):
    kernel_invocations = job["kernel_invocations"]
    mean_runtime = labmath.mean([ki.runtime for ki in kernel_invocations])

    avg = KernelInvocation(
        kernel_invocations[0].name,
        round(labmath.mean([ki.global_size for ki in kernel_invocations])),
        round(labmath.mean([ki.local_size for ki in kernel_invocations])),
        mean_runtime,
        transfer=round(labmath.mean([ki.transfer for ki in kernel_invocations])),
        dataset=kernel_invocations[0].dataset)
    avg.n = len(kernel_invocations)
    avg.ci = labmath.confinterval([ki.runtime for ki in kernel_invocations],
                                  array_mean=mean_runtime)[1] - mean_runtime
    return avg


def dump_csv(path, kernel_invocations):
    print("writing '{}'".format(path))
    with open(path, "w") as outfile:
        print(
            "benchmark",
            "dataset",
            "n",
            "transfer",
            "global_size",
            "local_size",
            "runtime",
            "ci",
            sep=",", file=outfile)
        for ki in kernel_invocations:
            print(ki.name,
                  ki.dataset,
                  ki.n,
                  ki.transfer,
                  ki.global_size,
                  ki.local_size,
                  "{:.6f}".format(ki.runtime),
                  "{:.6f}".format(ki.ci),
                  sep=",", file=outfile)


def dir2features(logdir, out=sys.stdout, metaout=sys.stderr):
    """
    Directory structure:

      <logdir>
      <logdir>/<iteration>
      <logdir>/<iteration>/<device>
      <logdir>/<iteration>/<device>/<log>
    """
    runs = fs.ls(logdir, abspaths=True)
    devices = fs.ls(runs[0])
    devlogs = get_device_logs(logdir)
    nlogs = sum([len(x) for x in devlogs.values()])
    num_workers = round(cpu_count() * 1.5)

    # Build a job list. Pack all the data required to complete a job
    # into a single object so that a worker thread can operator on it.
    for device in devlogs:
        devtype = device_to_device_type(device)

        jobs = [{"path": path, "device": device, "devtype": devtype}
                for path in devlogs[device]]
        print("spawning", num_workers, "worker threads to process",
              len(jobs), "jobs for", device, "...")

        shuffle(jobs)
        with Pool(num_workers) as pool:
            results = pool.map(_log_worker, jobs)

            # flatten kernel invocations across files
            kernel_invocations = [item for sublist in results for item in sublist]

            r = defaultdict(list)
            for ki in kernel_invocations:
                key = ki.name + ki.dataset
                r[key].append(ki)

            jobs = [{"kernel_invocations": r[key]} for key in r.keys()]
            shuffle(jobs)
            kernel_invocations = pool.map(_log_reducer, jobs)

            dump_csv(fs.path(logdir, device + "-dynamic.csv"),
                     kernel_invocations)
