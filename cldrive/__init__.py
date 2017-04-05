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
import platform

from collections import namedtuple
from enum import Enum, auto
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
    RAND = auto()
    SEQ = auto()
    ZEROS = auto()
    ONES = auto()


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


def make_data(src: str, gsize: NDRange, data_generator: Generator) -> np.array:
    """
    Generate data for OpenCL kernels.

    Returns:
        np.array: The generated data.

    Raises:
        InputTypeError: If any of the input arguments are of incorrect type.
    """
    _assert_or_raise(isinstance(src, str), InputTypeError)
    _assert_or_raise(isinstance(gsize, NDRange), InputTypeError)
    _assert_or_raise(isinstance(data_generator, Generator), InputTypeError)

    return np.array([[0, 1, 2, 3]])


def run_kernel(src: str, gsize: NDRange, lsize: NDRange,
               data: np.array=None,
               data_generator: Generator=Generator.SEQ,
               env: OpenCLEnvironment=None) -> np.array:
    """
    Execute an OpenCL kernel.

    Inputs:
        src (str): The OpenCL kernel source.
        gsize (NDRange): Kernel global size.
        lsize (NDRange): Kernel local (workgroup) size.
        data (np.array, optional): Data for kernel arguments.
        data_generator (Generator, optional): Type of data generator to use.
            Ignored if `data` argument is not None.
        env (OpenCLEnvironment, optional): The OpenCL environment to run the
            kernel in. If not provided, one is created by calling make_env()
            with no arguments.

    Returns:
        np.array: The output values.

    Raises:
        InputTypeError: If any of the input arguments are of incorrect type.
    """
    _assert_or_raise(isinstance(src, str), InputTypeError)
    _assert_or_raise(isinstance(gsize, NDRange), InputTypeError)
    _assert_or_raise(isinstance(lsize, NDRange), InputTypeError)
    _assert_or_raise(isinstance(data_generator, Generator), InputTypeError)

    # ensure we're working with a numpy array
    if data is None:
        data = make_data(src=src, gsize=gsize, data_generator=data_generator)
    else:
        data = np.array(data)

    if env is None:
        env = make_env()
    else:
        _assert_or_raise(len(env) == 2, InputTypeError)
        _assert_or_raise(isinstance(env[0], cl.Context), InputTypeError)
        _assert_or_raise(isinstance(env[1], cl.CommandQueue), InputTypeError)
    ctx, queue = env

    # clear any existing tasks in the command queue:
    queue.flush()

    return data * 2
