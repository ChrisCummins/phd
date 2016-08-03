from __future__ import division
from __future__ import print_function

from functools import wraps
from random import randrange
from threading import Thread

import numpy as np
import pyopencl as cl
import os
import signal
import sys

import labm8
from labm8 import fs
from labm8 import math as labmath

import smith
from smith import clutil


class DriveException(smith.SmithException): pass
class ProgramBuildException(DriveException): pass
class OpenCLDriverException(DriveException): pass

class KernelException(DriveException): pass
class E_BAD_CODE(KernelException): pass
class E_BAD_DRIVER(KernelException): pass
class E_BAD_ARGS(E_BAD_DRIVER): pass
class E_BAD_PROFILE(E_BAD_DRIVER): pass
class E_TIMEOUT(KernelException): pass
class E_OUTPUTS_UNCHANGED(KernelException): pass
class E_INPUT_INSENSITIVE(KernelException): pass
class E_NONDETERMINISTIC(KernelException): pass


def timeout(seconds=30):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise E_TIMEOUT("Process didn't terminate after {} seconds"
                            .format(seconds))

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


@timeout(30)
def run_with_timeout(kernel, *kargs):
    try:
        event = kernel(*kargs)
    except Exception as e:
        raise E_BAD_DRIVER
    return get_elapsed(event)



def build_program(ctx, src, quiet=True):
    """
    Compile an OpenCL program.
    """
    if not quiet:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    try:
        return cl.Program(ctx, src).build()
    except cl.cffi_cl.RuntimeError as e:
        raise ProgramBuildException(e)


def create_buffer_arg(ctx, dtype, size, read=True, write=True):
    host_data = np.random.rand(*size).astype(dtype)

    if read and write:
        mflags = cl.mem_flags.READ_WRITE
    elif read:
        mflags = cl.mem_flags.READ_ONLY
    elif write:
        mflags = cl.mem_flags.WRITE_ONLY
    else:
        raise smith.InternalException("buffer must allow reads or writes!")

    dev_data = cl.Buffer(ctx, mflags | cl.mem_flags.COPY_HOST_PTR,
                         hostbuf=host_data)
    transfer = 2 * host_data.nbytes
    return dev_data, transfer, host_data


def create_const_arg(ctx, dtype, val=None):
    if val is None: val = np.random.random_sample()
    dev_data = dtype(val)
    # TODO: Device whether we want const value args to be considered a
    # 'transfer'. If not, then transfer=0.
    transfer = dev_data.nbytes
    return dev_data, transfer, val


def get_payload(ctx, kernel, global_size):
    mf = cl.mem_flags

    arg_a, sz_a, host_a = create_buffer_arg(ctx, np.float32, global_size)
    arg_b, sz_b, host_b = create_buffer_arg(ctx, np.float32, global_size)
    arg_c, sz_c, _ = create_const_arg(ctx, np.int32, 100 * 50 - 1)

    transfer = sz_a + sz_b + sz_c

    args = (arg_a, arg_b, arg_c)

    try:
        kernel.set_args(*args)
    except cl.cffi_cl.LogicError as e:
        raise E_BAD_ARGS(e)
    except TypeError as e:
        raise E_BAD_ARGS(e)

    return transfer, args, {0: host_a, 1: host_b}


def kernel_name(kernel):
    return kernel.get_info(cl.kernel_info.FUNCTION_NAME)


def get_elapsed(event):
    """
    Time delta between event submission and end, in milliseconds.
    """
    try:
        event.wait()
        tstart = event.get_profiling_info(cl.profiling_info.SUBMIT)
        tend = event.get_profiling_info(cl.profiling_info.END)
        return (tend - tstart) / 1000000
    except Exception:
        raise E_BAD_PROFILE


def run_kernel(ctx, queue, kernel, global_size, filename='none'):
    name = kernel_name(kernel)

    transfer,args,hostd = get_payload(ctx, kernel, global_size)

    device = ctx.get_info(cl.context_info.DEVICES)[0]
    wgsize = kernel.get_work_group_info(
        cl.kernel_work_group_info.WORK_GROUP_SIZE, device)

    nruns = 10
    runtimes = []
    for i in range(nruns):
        # Blocking execution while kernel executes.
        elapsed = run_with_timeout(kernel, queue, global_size, None, *args)

        # Copy data back to host.
        for i in hostd:
            dev_buffer = args[i]
            host_buffer = hostd[i]
            event = cl.enqueue_copy(queue, host_buffer, dev_buffer)
            elapsed += get_elapsed(event)
        runtimes.append(elapsed)

    mean = labmath.mean(runtimes)
    ci = labmath.confinterval(runtimes, array_mean=mean)[1] - mean
    print(fs.basename(filename), name, wgsize, transfer,
          round(mean, 6), round(ci, 6), sep=',')


def assert_device_type(device, devtype):
    actual = device.get_info(cl.device_info.TYPE)
    if actual != devtype:
        requested = cl.device_type.to_string(devtype)
        received = cl.device_type.to_string(actual)
        raise OpenCLDriverException("Device type '{}' does not match "
                                    "requested '{}'"
                                    .format(received, requested))


def init_opencl(devtype=cl.device_type.GPU):
    platforms = cl.get_platforms()
    try:
        ctx = cl.Context(
            dev_type=devtype,
            properties=[(cl.context_properties.PLATFORM, platforms[0])])
    except Exception as e:
        ctx = cl.create_some_context(interactive=False)
    device = ctx.get_info(cl.context_info.DEVICES)[0]
    assert_device_type(device, devtype)
    cqp = cl.command_queue_properties
    queue = cl.CommandQueue(ctx, properties=cqp.PROFILING_ENABLE)

    return ctx, queue


def drive(ctx, queue, src, global_size, devtype=cl.device_type.GPU,
          quiet=True, filename='none'):
    """
    Execute a single OpenCL kernel.
    """
    program = build_program(ctx, src, quiet=quiet)
    kernels = program.all_kernels()
    assert(len(kernels) == 1)
    run_kernel(ctx, queue, kernels[0], global_size,
               filename=filename)


def kernel(src, filename='none', devtype=cl.device_type.GPU,
           global_size=None, **driveopts):
    # If no size is given, pick one.
    if global_size is None:
        global_size = (2 ** randrange(4, 15),)

    try:
        ctx, queue = init_opencl(devtype=devtype)
    except Exception as e:
        raise OpenCLDriverException(e)

    try:
        drive(ctx, queue, src, global_size,
              filename=filename, **driveopts)
    except DriveException as e:
        print('{}:'.format(fs.basename(filename)), e, file=sys.stderr)


def file(path, **driveopts):
    with open(fs.path(path)) as infile:
        src = infile.read()
        for kernelsrc in clutil.get_cl_kernels(src):
            kernel(kernelsrc, filename=fs.path(path), **driveopts)


def directory(path, **driveopts):
    files = fs.ls(fs.path(path), abspaths=True, recursive=True)
    for path in [f for f in files if f.endswith('.cl')]:
        file(path, **driveopts)
