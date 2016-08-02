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


class DriveException(smith.SmithException): pass
class ProgramBuildException(DriveException): pass

class KernelException(DriveException): pass
class E_BAD_CODE(KernelException): pass
class E_BAD_ARGS(KernelException): pass
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
def run_with_timeout(kernel, *kargs, timeout=1):
    event = kernel(*kargs)
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


def get_arg_prop(kernel, idx, prop, to_string):
    name = cl.kernel_arg_info.to_string(prop)
    try:
        val = to_string(kernel.get_arg_info(idx, prop))
    except RuntimeError as e:
        val = 'FAIL'
    return (name, val)


def get_arg_props(kernel, idx):
    ki = cl.kernel_arg_info
    props = [
        (ki.ACCESS_QUALIFIER, cl.kernel_arg_access_qualifier.to_string),
        (ki.ADDRESS_QUALIFIER, cl.kernel_arg_address_qualifier.to_string),
        (ki.NAME, str),
        (ki.TYPE_NAME, str),
        (ki.TYPE_QUALIFIER, cl.kernel_arg_type_qualifier.to_string)
    ]
    return dict(get_arg_prop(kernel, idx, *x) for x in props)


def get_args_props(kernel):
    nargs = kernel.get_info(cl.kernel_info.NUM_ARGS)
    return [get_arg_props(kernel, i) for i in range(nargs)]


def args_from_kernel(kernel):
    return get_args_props(kernel)


def is_pointer(props):
    return props['TYPE_NAME'].endswith('*')


def placeholder_from_props(props):
    if is_pointer(props):
        print('pointer')
    else:
        print('value')

    return props


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


def get_params(ctx, kernel, global_size):
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

    device = ctx.get_info(cl.context_info.DEVICES)[0]
    wgi = cl.kernel_work_group_info
    wgsize = kernel.get_work_group_info(wgi.WORK_GROUP_SIZE, device)

    return wgsize, transfer, args, {0: host_a, 1: host_b}


def kernel_name(kernel):
    return kernel.get_info(cl.kernel_info.FUNCTION_NAME)


def get_elapsed(event):
    """
    Time delta between event submission and end, in milliseconds.
    """
    event.wait()
    tstart = event.get_profiling_info(cl.profiling_info.SUBMIT)
    tend = event.get_profiling_info(cl.profiling_info.END)
    return (tend - tstart) / 1000000


def flatten_wgsize(wgsize):
    return np.prod(wgsize)


def run_kernel(ctx, queue, kernel, global_size=None, filename='none'):
    name = kernel_name(kernel)

    if global_size is None:
        global_size = (2 ** randrange(4, 15),)
    wgsize,transfer,args,hostd = get_params(ctx, kernel, global_size)

    nruns = 10
    runtimes = []
    for i in range(nruns):
        # Blocking execution while kernel executes.
        elapsed = run_with_timeout(kernel, queue, global_size, None, *args)

        # Copy data back to host.
        for i in hostd:
            dev_buffer = args[i]
            host_buffer = hostd[i]
            event = cl.enqueue_copy(queue, dev_buffer, dev_buffer)
            elapsed += get_elapsed(event)
        runtimes.append(elapsed)

    mean = labmath.mean(runtimes)
    ci = labmath.confinterval(runtimes, array_mean=mean)[1] - mean
    print(fs.basename(filename), name, wgsize, transfer,
          '{0:.6f}'.format(mean), '{0:.6f}'.format(ci), sep=',')


def init_opencl():
    platforms = cl.get_platforms()
    try:
        ctx = cl.Context(
            dev_type=devtype,
            properties=[(cl.context_properties.PLATFORM, platforms[0])])
    except Exception as e:
        ctx = cl.create_some_context(interactive=False)
    # print("Device:", ctx.get_info(cl.context_info.DEVICES))
    cqp = cl.command_queue_properties
    queue = cl.CommandQueue(ctx, properties=cqp.PROFILING_ENABLE)

    return ctx, queue


def drive(src, devtype=cl.device_type.GPU, quiet=True,
          global_size=None, filename='none'):
    ctx, queue = init_opencl()
    program = build_program(ctx, src, quiet=quiet)

    for kernel in program.all_kernels():
        try:
            run_kernel(ctx, queue, kernel, global_size=global_size,
                       filename=filename)
        except KernelException as e:
            pass


def file(path, **kwargs):
    with open(fs.path(path)) as infile:
        src = infile.read()
        try:
            drive(src, filename=fs.path(path), **kwargs)
        except DriveException as e:
            print(e, file=sys.stderr)


def directory(path, **kwargs):
    for path in fs.ls(fs.path(path), abspaths=True, recursive=True):
        file(path, **kwargs)
