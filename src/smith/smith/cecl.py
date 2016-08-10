#
# libcecl utilities.
#
# WARNING: This code is custom tailored to one specific experimental
# setup and methodology. It is EXTREMELY fragile code!
#
from __future__ import division,absolute_import,print_function,unicode_literals

import editdistance
import os
import re
import six
import sys

from collections import defaultdict
from itertools import product
from multiprocessing import cpu_count, Pool

import labm8
from labm8 import fs

import smith
from smith import clutil


class CeclException(smith.SmithException): pass
class LogException(CeclException): pass
class NameInferenceException(CeclException): pass


def assert_device_type(expected, actual):
    if expected.lower() != actual.lower():
        raise LogException("expected device type '{}', found device type '{}'"
                           .format(expected, actual))


def parse_cecl_log(log, devtype=None):
    """
    Interpret and parse the output of a libcecl instrument kernel.
    """
    lines = []

    insrc = False
    srcbuf = ''
    with open(log) as infile:
        contents = infile.read()

    for line in contents.split('\n'):
        if line.strip() == 'BEGIN PROGRAM SOURCE':
            insrc = True
        elif line.strip() == 'END PROGRAM SOURCE':
            insrc = False
            kernels = [clutil.strip_attributes(x)
                       for x in clutil.get_cl_kernels(srcbuf)]
            names = [x.split()[2].split('(')[0] for x in kernels]
            lines[-1].append(dict(zip(names, kernels)))
            srcbuf = ''
        elif insrc:
            srcbuf += line + "\n"
        else:
            components = [x.strip() for x in line.split(';')]
            if devtype and components[0] == "clCreateCommandQueue":
                try:
                    assert_device_type(devtype, components[1])
                except LogException as e:
                    raiseLogException(log + ": " + str(e))
            elif components[0]:
                lines.append(components)
    return lines


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


def get_kernels(parsed):
    compiled_k = {}     # maps function names to implementations
    enqueued_k = set()  # store all functions which actually get executed

    for line in parsed:
        if line[0] == 'clCreateProgramWithSource':
            for function_name,source in six.iteritems(line[1]):
                compiled_k[function_name] = source
        elif line[0] == 'clEnqueueNDRangeKernel':
            function_name = line[1]
            enqueued_k.add(function_name)
        elif line[0] == 'clEnqueueTask':
            function_name = line[1]
            print("TASK", function_name)
            enqueued_k.add(function_name)

    # Check that we have source code for all enqueued kernels.
    undefined = []
    for kernel in enqueued_k:
        if kernel not in compiled_k:
            undefined.append(kernel)
    if len(undefined):
        print("undefined kernels:",
              ", ".join("'{}'".format(x) for x in undefined), file=sys.stderr)

    # Remove any kernels which are not used in the source code.
    # unused = []
    for key in list(compiled_k.keys()):
        if key not in enqueued_k:
            # unused.append(key)
            compiled_k.pop(key)
    # if len(unused):
    #     print("unused kernels:", ', '.join("'{}'".format(x) for x in unused))

    return compiled_k


def get_transfers(parsed):
    transfers = defaultdict(list) # maps buffer names to (size,elapsed) tuples

    for line in parsed:
        if (line[0] == 'clEnqueueReadBuffer' or
            line[0] == 'clEnqueueWriteBuffer'):
            buf, size, elapsed = line[1:]
            transfers[buf].append((int(size), float(elapsed)))

    return transfers


def get_kernel_args(parsed, kernels):
    kernel_args = defaultdict(list)

    for line in parsed:
        if line[0] == "clSetKernelArg":
            kernel_name, index, size, name = line[1:]
            # Only record the kernel argument if it is actually
            # enqueued.
            if kernel_name in kernels:
                kernel_args[kernel_name].append(name)

    return kernel_args


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
    parsed = parse_cecl_log(log, devtype=devtype)

    kernels = get_kernels(parsed)
    transfers = get_transfers(parsed)
    kernel_args = get_kernel_args(parsed, kernels)

    # for transfer in transfers.keys():
    #     print('-'.join((benchmark, dataset)), transfer,
    #           len(transfers[transfer]))

    # for kernel in kernels.keys():
    #     print('-'.join((benchmark, dataset, kernel)))


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
    rundirs = fs.ls(logdir, abspaths=True)
    devices = [fs.basename(x) for x in fs.ls(rundirs[0])]

    for rundir,device in product(rundirs, devices):
        devlogs[device] += sorted(
            [x for x in fs.ls(fs.path(rundir, device), abspaths=True)
             if fs.isfile(x)])

    # The number of files in the logdir:
    nfiles = len([x for x in fs.ls(logdir, abspaths=True, recursive=True)
                  if fs.isfile(x)])
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

    print("Common subset:")
    print("    # files:", nlogs, "({:.1f}%)".format((nlogs / nfiles) * 100))
    print("    # common:", len(common))
    print("    # devices:", len(devices), "({})".format(
        ", ".join(["{}: {}".format(x, len(devlogs[x])) for x in devlogs])))

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


def dir2features_worker(job):
    process_cecl_log(job["path"], devtype=job["devtype"])


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

    jobs = []
    for device in devlogs:
        devtype = device_to_device_type(device)
        for path in devlogs[device]:
            jobs.append({"path": path, "devtype": devtype})

    num_workers = round(cpu_count() * 1.5)
    with Pool(num_workers) as pool:
        print('\nspawning', num_workers, 'worker threads to process',
              len(jobs), 'jobs ...')
        pool.map(dir2features_worker, jobs)
