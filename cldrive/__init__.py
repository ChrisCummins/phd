# Copyright (C) 2017 Chris Cummins.
#
# This file is part of cldrive.
#
# Cldrive is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Cldrive is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Run arbitrary OpenCL kernels.
"""
import math
import os
import platform

from collections import namedtuple
from enum import Enum, auto
from functools import partial
from typing import List

import numpy as np
import pyopencl as cl


class CLdriveError(Exception):
    """ Base error type. """
    pass

class InputTypeError(ValueError, CLdriveError):
    """ Raised if an input is of incorrect type. """
    pass


class OpenCLDeviceNotFound(LookupError, CLdriveError):
    """ Raised if no matching OpenCL device is found. """
    pass


class ParseError(ValueError, CLdriveError):
    """ Raised if parsing OpenCL source fails. """
    pass


class KernelArgError(ValueError, CLdriveError):
    """ Raised if kernel argument processing fails. """
    pass

from cldrive import payload


NDRange = namedtuple('NDRange', ['x', 'y', 'z'])
OpenCLEnvironment = namedtuple('OpenCLEnvironment', ['ctx', 'queue'])


class Generator(Enum):
    # We wrap functions in a partial so that they are interpreted as attributes
    # rather than methods. See: http://stackoverflow.com/a/40339397
    RAND = partial(np.random.rand)
    SEQ = partial(np.arange)
    ZEROS = partial(np.zeros)
    ONES = partial(np.ones)

    def __call__(self, numpy_type: np.dtype, *args, **kwargs):
        """ generate arrays of data """
        return self.value(*args, **kwargs).astype(numpy_type)

    @staticmethod
    def from_str(string) -> Generator:
        if string == "rand":
            return Generator.RAND
        elif string == "seq":
            return Generator.SEQ
        elif string == "zeros":
            return Generator.ZEROS
        elif string == "ones":
            return Generator.ONES
        else:
            raise InputTypeError


def _assert_or_raise(stmt: bool, exception=CLdriveError,
                     *exception_args, **exception_kwargs) -> None:
    """
    If the statement is false, raise the given exception.
    """
    if not stmt:
        raise exception(*exception_args, **exception_kwargs)


def make_env(platform_id: int=None, device_id: int=None,
             devtype=None, queue_flags=0) -> OpenCLEnvironment:
    """
    Create an OpenCL context and device queue.

    Iterates over the available OpenCL platforms and devices looking for a
    device matching the requested platform ID, or platform and device ID, or
    device type. Constructs and returns an OpenCL context and queue for the
    matching device. Note that OpenCL profiling is enabled.

    Arguments:
        platform_id (int, optional): OpenCL Platform ID. If not provided, any
            available platform may be used.
        device_id (int, optional): OpenCL Device ID. If not provided, any
            available device may be used. Requires that platform_id is set.
        devtype (pyopencl.device_type, optional): OpenCL device type.
            If not specified, the first available device will be used.
        queue_flags (cl.command_queue_properties, optional): Bitfield of
            OpenCL queue constructor options.

    Returns:
        OpenCLEnvironment: A named tuple consisting of an OpenCL context and
            device queue.

    Raises:
        ValueError: If device_id is set, but not platform_id.
        OpenCLDeviceNotFound: If no matching type found.
    """
    def device_type_matches(device, devtype) -> bool:
        """ check that device type matches """
        if devtype:
            actual_devtype = device.get_info(cl.device_info.TYPE)
            return actual_devtype == devtype
        else:  # no devtype to match against
            return True

    # get list of platforms to iterate over. If platform ID is provided, use
    # only that platform.
    if platform_id is None:
        platforms = cl.get_platforms()
    else:
        try:
            platforms = [cl.get_platforms()[platform_id]]
        except IndexError:
            raise OpenCLDeviceNotFound(f"No platform for id={platform_id}")

    for platform in platforms:
        ctx = cl.Context(
            properties=[(cl.context_properties.PLATFORM, platform)])

        # get list of devices to iterate over. If device ID is provided, use
        # only that device. Else, take any device which matches devtype
        if device_id is None:
            devices = ctx.get_info(cl.context_info.DEVICES)
        else:
            _assert_or_raise(platform_id is not None, ValueError)
            try:
                devices = [ctx.get_info(cl.context_info.DEVICES)[device_id]]
            except IndexError:
                raise OpenCLDeviceNotFound(f"No device for id={device_id}")

        devices = [d for d in devices if device_type_matches(d, devtype)]

        if len(devices):
            queue = cl.CommandQueue(ctx, properties=queue_flags)
            return OpenCLEnvironment(ctx=ctx, queue=queue)

    raise OpenCLDeviceNotFound("Could not find a suitable device")


def extract_args(src: str) -> List[payload.KernelArg]:
    """
    Extract kernel arguments for an OpenCL kernel.

    Returns:
        List[payload.KernelArg]: A list of the kernel's arguments, in order.

    Raises:
        ParseError: If the source contains more than one kernel definition,
            or if any of the kernel's parameters cannot be determined.
        KernelArgError: If any of the kernel's parameters are invalid or
            not supported.
    """
    ast = payload.parse(src)
    visitor = payload.ArgumentExtractor()
    visitor.visit(ast)
    return visitor.args


def make_data(src: str, size: int, data_generator: Generator,
              scalar_val: float=None) -> np.array:
    """
    Generate data for OpenCL kernels.

    Creates a numpy array for each OpenCL argument, except arguments with the
    'local' qualifier, since those are instantiated.

    Returns:
        np.array: The generated data.

    Raises:
        InputTypeError: If any of the input arguments are of incorrect type.
    """
    # check the input types
    _assert_or_raise(isinstance(src, str), InputTypeError)
    _assert_or_raise(isinstance(data_generator, Generator), InputTypeError)

    if scalar_val is None:
        scalar_val = size

    args = extract_args(src)
    args_with_inputs = [arg for arg in args if arg.has_host_input]

    data = []
    for arg in args_with_inputs:
        if arg.is_pointer:
            argdata = data_generator(arg.numpy_type, size * arg.vector_width)
        else:
            # scalar values are still arrays, so e.g. 'float4' is an array of
            # 4 floats. Each component of a scalar value is the flattened
            # global size, e.g. with gsize (32,2,1), scalar arugments have the
            # value 32 * 2 * 1 = 64.
            scalar_val = [scalar_val] * arg.vector_width
            argdata = np.array(scalar_val).astype(arg.numpy_type)

        data.append(argdata)

    return np.array(data)


def run_kernel(src: str, gsize: NDRange, lsize: NDRange,
               data: np.array=None, buf_scale: float=1.0,
               data_generator: Generator=Generator.SEQ, timeout: float=-1,
               env: OpenCLEnvironment=None, debug: bool=False) -> np.array:
    """
    Execute an OpenCL kernel.

    Inputs:
        src (str): The OpenCL kernel source.
        gsize (NDRange): Kernel global size.
        lsize (NDRange): Kernel local (workgroup) size.
        data (np.array, optional): Data for kernel arguments.
        buf_scale (float, optional): Scaling factor for multiplying global
            size dimensions when generating data.
        data_generator (Generator, optional): Type of data generator to use.
            Ignored if `data` argument is not None.
        timeout (float, optional): Cancel execution if it has not completed
            after this many seconds. A value <= 0 means never time out.
        env (OpenCLEnvironment, optional): The OpenCL environment to run the
            kernel in. If not provided, one is created by calling make_env()
            with no arguments.
        debug(bool, optional): If true, silence the OpenCL compiler compiler.

    Returns:
        np.array: The output values.

    Raises:
        InputTypeError: If any of the input arguments are of incorrect type.

    TODO:
        * Implement timeout.
        * Implement Optimizations on or off.
    """
    # check our input types
    _assert_or_raise(isinstance(src, str), InputTypeError)
    _assert_or_raise(len(gsize) == 3, InputTypeError)
    _assert_or_raise(len(lsize) == 3, InputTypeError)
    _assert_or_raise(isinstance(data_generator, Generator), InputTypeError)
    gsize = NDRange(*gsize)
    lsize = NDRange(*lsize)

    # setup our OpenCL environment as required
    if env is None:
        env = make_env()
    else:
        _assert_or_raise(len(env) == 2, InputTypeError)
        _assert_or_raise(isinstance(env[0], cl.Context), InputTypeError)
        _assert_or_raise(isinstance(env[1], cl.CommandQueue), InputTypeError)
    ctx, queue = env

    # scalar_gsize is the product of the global NDRange.
    # buf_size is scaled relative to the global size in each dimension.
    scalar_gsize: int = 1
    buf_size: int = 1
    for x in gsize:
        scalar_gsize *= x
        buf_size *= x * buf_scale if x > 1 else x
    buf_size = math.ceil(buf_size)

    # generate data and ensure we're working with a numpy array
    if data is None:
        data = make_data(src=src, size=buf_size, scalar_val=scalar_gsize,
                         data_generator=data_generator)
    else:
        data = np.array(data, copy=True)

    # get the indices of input arguments
    args = extract_args(src)
    data_indices = [i for i, arg in enumerate(args) if arg.has_host_input]

    # sanity check that there are enough the correct number of inputs
    _assert_or_raise(len(data_indices) == len(data), InputTypeError,
                     "Incorrect number of inputs provided")

    ArgTuple = namedtuple('ArgTuple', ['kernelarg', 'hostdata', 'devdata'])
    argtuples = []
    for i, arg in enumerate(args):
        if isinstance(arg, payload.GlobalBufferArg):
            hostdata = data[i]
        else:
            hostdata = None

        if isinstance(arg, payload.LocalBufferArg):
            nbytes = buf_size * arg.vector_width * arg.numpy_type.itemsize
            devdata = cl.LocalMemory(nbytes)
        elif isinstance(arg, payload.GlobalBufferArg):
            assert(i in data_indices)  # sanity check
            data_indices.remove(i)

            # determine flags to pass to OpenCL buffer creation:
            flags = cl.mem_flags.COPY_HOST_PTR
            if arg.is_const:
                flags |= cl.mem_flags.READ_ONLY
            else:
                flags |= cl.mem_flags.READ_WRITE

            devdata = cl.Buffer(ctx, flags, hostbuf=hostdata)
        elif isinstance(arg, payload.ScalarArg):
            assert(i in data_indices)
            data_indices.remove(i)

            devdata = data[i]
        argtuples.append(ArgTuple(kernelarg=arg, hostdata=hostdata,
                                  devdata=devdata))
    assert(not len(data_indices))

    # clear any existing tasks in the command queue:
    queue.flush()

    # compile the program
    if debug:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    else:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
    program = cl.Program(ctx, src).build()
    kernel = program.all_kernels()[0]

    # copy data to device
    for arg in argtuples:
        if arg.hostdata is not None:
            cl.enqueue_copy(queue, arg.devdata, arg.hostdata,
                            is_blocking=False)

    kernel_args = [t.devdata for t in argtuples]

    # set the kernel arguments
    kernel.set_args(*kernel_args)

    # run the kernel
    kernel(queue, gsize, lsize, *kernel_args)

    for arg in argtuples:
        if arg.hostdata is not None and not arg.kernelarg.is_const:
            cl.enqueue_copy(queue, arg.hostdata, arg.devdata,
                            is_blocking=False)

    # wait for queue to finish
    queue.flush()

    return data
