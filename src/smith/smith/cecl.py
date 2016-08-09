#
# libcecl utilities.
#
from __future__ import division,absolute_import,print_function,unicode_literals

import os
import six
import sys

import editdistance

import labm8
from labm8 import fs

import smith
from smith import clutil


class NameInferenceException(smith.SmithException): pass

KNOWN_KERNEL_MAPPINGS = {
    "k_cs": "create_seq",
    "kernel_likelihood": "likelihood_kernel",
    "kernel_find_index": "find_index_kernel"
}


def log2features(log, out=sys.stdout, metaout=sys.stderr):
    return {
        "benchmark": "",
        "dataset": "",
        "comp": "",
        "rational": "",
        "mem": "",
        "localmem": "",
        "coalesced": "",
        "atomic": "",
        "transfer": "",
        "wgsize": "",
        "F1:transfer/(comp+mem)": "",
        "F2:coalesced/mem": "",
        "F3:(localmem/mem)*avgws": "",
        "F4:comp/mem": "",
        "runtime": "",
        "run": ""
    }


def parse_cecl_log(log):
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
            kernels = clutil.get_cl_kernels(srcbuf)
            names = [x.split()[2].split('(')[0] for x in kernels]
            lines[-1].append(dict(zip(names, kernels)))
            srcbuf = ''
        elif insrc:
            srcbuf += line
        else:
            components = [x.strip() for x in line.split(';')]
            if components[0]:
                lines.append(components)
    return lines


def get_kernels(parsed):
    compiled_k = {}  # maps source names to implementations
    enqueued_k = {}  # maps variable names to source names

    for line in parsed:
        if line[0] == 'clCreateProgramWithSource':
            for k,v in six.iteritems(line[1]):
                compiled_k[k] = v
        elif line[0] == 'clEnqueueNDRangeKernel':
            variable_name = line[1]

            # First, look-up the global mapping table to find a match.
            mapping = KNOWN_KERNEL_MAPPINGS.get(variable_name, None)
            if mapping:
                enqueued_k[line[1]] = mapping
                continue

            # Look-up the names of compiled kernels to see if has been
            # declared. Maybe the variable and source names are the
            # same?
            mapping = compiled_k.get(variable_name, None)
            if mapping:
                enqueued_k[line[1]] = variable_name
                continue

            # If all else fails, we'll try and infer the source name
            # by looking for a unique minimum edit distance between
            # variable and kernel name.
            distances = [(x, editdistance.eval(variable_name, x))
                         for x in compiled_k.keys()]
            mindistance = min(x[1] for x in distances)
            minval = [x for x in distances if x[1] == mindistance]
            # If more than one value shares the same edit distance,
            # then fail.
            if len(minval) > 1:
                print("failed to infer source name for kernel variable "
                      "'{}'. Found {} candidates: {} with distance {}"
                      .format(
                          variable_name, len(minval),
                          ', '.join(["'{}'".format(y[0]) for y in minval]),
                          mindistance))
            # Success! We have inferred the source name from the
            # variable name:
            else:
                enqueued_k[line[1]] = minval[0][0]

    print("Inferred kernel to source name mappings:")
    for k,v in six.iteritems(enqueued_k):
        if k != v:
            print(k, v, sep=' -> ')

    # Remove any kernels which are not used in the source code.
    unused = []
    for key in list(compiled_k.keys()):
        if key not in enqueued_k.values():
            unused.append(key)
            compiled_k.pop(key)
    if len(unused):
        print("unused kernels:", ', '.join("'{}'".format(x) for x in unused))

    return compiled_k, enqueued_k


def process_cecl_log(log):
    parsed = parse_cecl_log(log)
    kernels, kernel_mappings = get_kernels(parsed)


def dir2features(log, out=sys.stdout, metaout=sys.stderr):
    runs = fs.ls(log, abspaths=True)
    devices = fs.ls(runs[0])
    files = [x for x in fs.ls(log, abspaths=True, recursive=True)
             if fs.isfile(x)]

    print("summarising", len(files), "logs:", file=metaout)
    print("   # runs:", len(runs), file=metaout)
    print("   # devices:", len(devices), file=metaout)

    for path in files:
        process_cecl_log(path)
