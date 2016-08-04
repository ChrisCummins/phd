from __future__ import division
from __future__ import print_function

from copy import deepcopy
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
from smith import config as cfg
from smith import clutil


class DriveException(smith.SmithException): pass
class OpenCLDriverException(DriveException): pass
class OpenCLNotSupported(OpenCLDriverException): pass

class KernelDriverException(DriveException): pass

class E_BAD_CODE(KernelDriverException): pass
class E_UGLY_CODE(KernelDriverException): pass

class E_BAD_DRIVER(KernelDriverException): pass
class E_BAD_ARGS(E_BAD_DRIVER): pass
class E_BAD_PROFILE(E_BAD_DRIVER): pass

class E_TIMEOUT(E_BAD_CODE): pass

class E_INPUT_INSENSITIVE(E_UGLY_CODE): pass
class E_NO_OUTPUTS(E_UGLY_CODE): pass
class E_NONDETERMINISTIC(E_UGLY_CODE): pass


def init_opencl(devtype=cl.device_type.GPU):
    if not cfg.host_has_opencl():
        raise OpenCLNotSupported

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


def timeout(seconds=30):
    """
    Returns a decorator for executing a process with a timeout.
    """
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
    """
    Run an OpenCL kernel, and block until completed.

    Arguments:

        kernel:  A pyopencl.Kernel instance
        *kargs:  Arguments to invoke kernel with

    Returns:

        Elapsed time (see get_elapsed()).

    Raises:

        E_TIMEOUT: If execution does not complete within given time.
    """
    event = kernel(*kargs)
    return get_elapsed(event)


class KernelDriver(object):
    """
    OpenCL Kernel driver. Drives a single OpenCL kernel.

    Arguments:

        ctx:  pyopencl context.
        queue: OpenCL queue for context.
        source: String kernel source.

    Raises:

        E_BAD_CODE: If program doesn't compile.
        E_UGLY_CODE: If program contains multiple kernels.
    """
    def __init__(self, ctx, source, source_path='<stdin>'):
        self._ctx = ctx
        self._src = str(source)
        self._program = KernelDriver.build_program(self._ctx, self._src)
        self._prototype = clutil.KernelPrototype.from_source(self._src)

        kernels = self._program.all_kernels()
        if len(kernels) != 1:
            raise E_UGLY_CODE
        self._kernel = kernels[0]
        self._name = self._kernel.get_info(cl.kernel_info.FUNCTION_NAME)

        # Profiling stats
        self._wgsizes = []
        self._transfers = []
        self._runtimes = []

    @timeout(30)
    def __call__(queue, payload):
        output = deepcopy(payload)

        kargs = output.kargs
        self.kernel(*kargs)
        output.device_to_host(queue)

        return output

    def __repr__(self):
        return self.source

    @property
    def context(self): return self._ctx

    @property
    def source(self): return self._src

    @property
    def program(self): return self._program

    @property
    def prototype(self): return self._prototype

    @property
    def kernel(self): return self._kernel

    @property
    def name(self): return self._name

    @property
    def wgsizes(self): return self._wgsizes

    @property
    def transfers(self): return self._transfers

    @property
    def runtimes(self): return self._runtimes

    @staticmethod
    def build_program(ctx, src, quiet=True):
        """
        Compile an OpenCL program.
        """
        if not quiet:
            os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        try:
            return cl.Program(ctx, src).build()
        except cl.cffi_cl.RuntimeError as e:
            raise E_BAD_CODE(e)


class KernelPayload(object):
    def __init__(self, ctx, args):
        self._ctx = ctx
        self._args = args

    def __deepcopy__(self):
        print('DEEPDEEP', file=sys.stderr)

    def __eq__(self, other):
        """
        """
        if self.context != other.context:
            return False

        if len(self.args) != len(other.args):
            return False

        for x,y in zip(self.args, other.args):
            if type(x) != type(y):
                return False
            if x.hostdata is None:
                if x.devdata != y.devdata:
                    return False
            else:
                if any([e1 != e2 for e1,e2 in zip(x.hostdata,y.hostdata)]):
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def device_to_host(self, queue):
        elapsed = 0
        return elapsed

    @property
    def context(self): return self._ctx

    @property
    def args(self): return self._args

    @property
    def kargs(self): return [a.devdata for a in self._args]

    @staticmethod
    def _create_payload(nparray, driver, size):
        args = [clutil.KernelArg(arg.string) for arg in driver.prototype.args]

        for arg in args:
            dtype = arg.numpy_type
            arg.hostdata = None
            if arg.is_pointer:
                veclength = size * arg.vector_width
                arg.hostdata = nparray(veclength)
                flags = cl.mem_flags.COPY_HOST_PTR
                if arg.is_const:
                    flags |= cl.mem_flags.READ_ONLY
                else:
                    flags |= cl.mem_flags.READ_WRITE
                arg.devdata = cl.Buffer(
                    driver.context, flags, hostbuf=arg.hostdata)
            else:
                arg.devdata = dtype(size)

        return KernelPayload(driver.context, args)

    @staticmethod
    def create_sequential(*args, **kwargs):
        return KernelPayload._create_payload(np.arange, *args, **kwargs)

    @staticmethod
    def create_random(*args, **kwargs):
        return KernelPayload._create_payload(np.random.rand, *args, **kwargs)


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


def run_kernel(ctx, queue, kernel, size, filename='none'):
    name = kernel_name(kernel)

    transfer,args,hostd = get_payload(ctx, kernel, size)

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


def drive(ctx, queue, src, size, devtype=cl.device_type.GPU,
          quiet=True, filename='none'):
    """
    Execute a single OpenCL kernel.
    """
    program = build_program(ctx, src, quiet=quiet)
    kernels = program.all_kernels()
    assert(len(kernels) == 1)
    run_kernel(ctx, queue, kernels[0], global_size,
               filename=filename)


def kernel(src, filename='<stdin>', devtype=cl.device_type.GPU,
           size=None, file=sys.stdout, metaout=sys.stderr, **driveopts):
    """
    Drive a kernel
    """
    def assert_constraint(constraint, err=DriveException):
        if not constraint:
            raise err

    # If no size is given, pick one.
    if size is None:
        size = 2 ** randrange(4, 15)

    try:
        ctx, queue = init_opencl(devtype=devtype)
    except Exception as e:
        raise OpenCLDriverException(e)

    # Create driver.
    driver = KernelDriver(ctx, src, source_path=filename)

    try:
        # Create payloads.
        A1in = KernelPayload.create_sequential(driver, size)
        A2in = deepcopy(A1in)

        B1in = KernelPayload.create_random(driver, size)
        B2in = deepcopy(B1in)

        # Input constraints.
        assert_constraint(A1in == A2in, E_BAD_DRIVER)
        assert_constraint(B1in == B2in, E_BAD_DRIVER)
        assert_constraint(A1in != B1in, E_BAD_DRIVER)

        A1out = driver(A1in, queue)
        B1out = driver(B1in, queue)
        A2out = driver(A2in, queue)
        B2out = driver(B2in, queue)

        # outputs must be consistent across runs:
        assert_constraint(A1out == A2out, E_NONDETERMINISTIC)
        assert_constraint(B1out == B2out, E_NONDETERMINISTIC)

        # outputs must depend on inputs:
        assert_constraint(A1out != B1out, E_INPUT_INSENSITIVE)

        # outputs must be different from inputs:
        assert_constraint(A1in != A1out, E_NO_OUTPUTS)
        assert_constraint(B1in != B1out, E_NO_OUTPUTS)

        wgsize = round(labmath.mean(driver.wgsizes))
        transfer = round(labmath.mean(driver.transfers))
        mean = round(labmath.mean(driver.runtimes), 6)
        ci = round(labmath.confinternval(driver.runtimes, array_mean=mean), 6)

        print(filename, wgsize, transfer, mean, ci, sep=',', file=file)
    except DriveException as e:
        print('-'.join((filename, driver.name)), e, sep=',', file=metaout)
        raise e


def file(path, **driveopts):
    with open(fs.path(path)) as infile:
        src = infile.read()
        for kernelsrc in clutil.get_cl_kernels(src):
            kernel(kernelsrc, filename=fs.path(path), **driveopts)


def directory(path, **driveopts):
    files = fs.ls(fs.path(path), abspaths=True, recursive=True)
    for path in [f for f in files if f.endswith('.cl')]:
        file(path, **driveopts)
