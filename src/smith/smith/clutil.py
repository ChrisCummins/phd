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


class OpenCLUtilException(smith.SmithException): pass
class PrototypeException(OpenCLUtilException): pass

class KernelArg(object):
    """
    OpenCL Kernel Argument.

    Requires source code to have been pre-processed.
    """
    def __init__(self, string):
        self._string = string.strip()
        self._components = self._string.split(' ')

    @property
    def name(self):
        return self._components[-1]

    @property
    def type(self):
        return self._components[-2]

    @property
    def qualifiers(self):
        return self._components[:-2]

    @property
    def is_pointer(self):
        return self.type[-1] == '*'

    @property
    def is_global(self):
        try:
            return self._is_global
        except AttributeError:  # set
            self._is_global = (True if
                               '__global' in self.qualifiers or
                               'global' in self.qualifiers
                               else False)
            return self._is_global

    @property
    def is_local(self):
        try:
            return self._is_local
        except AttributeError:  # set
            self._is_local = (True if
                              '__local' in self.qualifiers or
                              'local' in self.qualifiers
                              else False)
            return self._is_local

    def __repr__(self):
        return self._string


class KernelPrototype(object):
    """
    OpenCL Kernel Prototype.

    Requires source code to have been pre-processed.
    """
    def __init__(self, string):
        self._string = string.strip()
        if not self._string.startswith('__kernel void '):
            raise PrototypeException('malformed prototype', self._string)

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:  # set
            idx_start = len('__kernel void ')
            idx_end = self._string.find('(')
            self._name = self._string[idx_start:idx_end]
            return self._name

    @property
    def args(self):
        try:
            return self._args
        except AttributeError:
            idx_open_brace = self._string.find('(')
            idx_close_brace = self._string.find(')')
            inner_brace = self._string[idx_open_brace + 1:idx_close_brace]
            self._args = [KernelArg(x) for x in inner_brace.split(',')]
            return self._args

    def __repr__(self):
        return self._string

    @staticmethod
    def from_source(src):
        return extract_prototype(src)


def get_cl_kernel(src, start_idx):
    """
    Return the OpenCL kernel.

    :param src: OpenCL source.
    :return: Kernel implementation.
    """
    i = src.find('{', start_idx) + 1
    d = 1  # depth
    while i < len(src) and d > 0:
        if src[i] == '{':
            d += 1
        elif src[i] == '}':
            d -= 1
        i += 1
    return src[start_idx:i]


def get_cl_kernels(src):
    """
    Return OpenCL kernels.

    :param src: OpenCL source.
    :return: Kernel implementations.
    """
    idxs = smith.get_substring_idxs('__kernel void ', src)
    kernels = [get_cl_kernel(src, i) for i in idxs]
    return kernels


def extract_prototype(src):
    """
    Extract OpenCL kernel prototype from rewritten file.

    :param src: OpenCL kernel source.
    :return: KernelPrototype object instance.
    """
    idxs = smith.get_substring_idxs('__kernel void ', src)
    if len(idxs) != 1:
        raise PrototypeException("Invalid number of kernels found: {}"
                                 .format(len(idxs)))
    src = get_cl_kernel(src, idxs[0])

    try:
        index = src.index('{') + 1
        prototype = src[:index]
    except ValueError:
        raise PrototypeException("malformed seed")

    return KernelPrototype(prototype)
