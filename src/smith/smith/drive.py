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

class E_NON_TERMINATING(E_BAD_CODE): pass

class E_INPUT_INSENSITIVE(E_UGLY_CODE): pass
class E_NO_OUTPUTS(E_UGLY_CODE): pass
class E_NONDETERMINISTIC(E_UGLY_CODE): pass


def assert_device_type(device, devtype):
    """
    Check that device type matches.

    Raises:

        OpenCLDriverException: If device type does not match argument.
    """
    actual = device.get_info(cl.device_info.TYPE)
    if actual != devtype:
        requested = cl.device_type.to_string(devtype)
        received = cl.device_type.to_string(actual)
        raise OpenCLDriverException("Device type '{}' does not match "
                                    "requested '{}'"
                                    .format(received, requested))


def init_opencl(devtype=cl.device_type.GPU, queue_flags=0):
    """
    Initialise an OpenCL context with some command queue.

    Raises:

        OpenCLNotSupported: If host does not support OpenCL.
        OpenCLDriverException: In case of error.
    """
    if not cfg.host_has_opencl():
        raise OpenCLNotSupported

    platforms = cl.get_platforms()
    try:
        ctx = cl.Context(
            dev_type=devtype,
            properties=[(cl.context_properties.PLATFORM, platforms[0])])
    except Exception:
        try:
            ctx = cl.create_some_context(interactive=False)
        except Exception as e:
            raise OpenCLDriverException(e)

    # Check that device type is what we expected:
    device = ctx.get_info(cl.context_info.DEVICES)[0]
    assert_device_type(device, devtype)
    queue_flags |= cl.command_queue_properties.PROFILING_ENABLE
    queue = cl.CommandQueue(ctx, properties=queue_flags)

    return ctx, queue


def get_event_time(event):
    """
    Block until OpenCL event has completed and return time delta
    between event submission and end, in milliseconds.

    Raises:

        E_BAD_PROFILE: In case of error.
    """
    try:
        event.wait()
        tstart = event.get_profiling_info(cl.profiling_info.SUBMIT)
        tend = event.get_profiling_info(cl.profiling_info.END)
        return (tend - tstart) / 1000000
    except Exception:
        # Possible exceptions:
        #
        #  pyopencl.cffi_cl.RuntimeError: clwaitforevents failed: OUT_OF_RESOURCES
        #
        raise E_BAD_PROFILE


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
    def __call__(self, queue, payload, timeout=30):
        elapsed = 0
        output = deepcopy(payload)

        kargs = output.kargs

        # Run kernel and get time.
        elapsed += output.host_to_device(queue)

        # Try setting the kernel arguments.
        try:
            self.kernel.set_args(*kargs)
        except cl.cffi_cl.LogicError as e:
            raise E_BAD_ARGS(e)
        except TypeError as e:
            raise E_BAD_ARGS(e)

        # Execute kernel
        event = self.kernel(queue, output.ndrange, None, *kargs)
        elapsed += get_event_time(event)

        # Copy data back to host and get time.
        elapsed += output.device_to_host(queue)

        # Record workgroup size.
        device = self.context.get_info(cl.context_info.DEVICES)[0]
        wgsize = self.kernel.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
        self.wgsizes.append(wgsize)

        # Record runtime.
        self.runtimes.append(elapsed)

        # Record transfers.
        self.transfers.append(payload.transfersize)

        return output

    def __repr__(self):
        return self.source

    def validate(self, size=16):
        def assert_constraint(constraint, err=DriveException):
            if not constraint:
                raise err

        # TODO:
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
    def __init__(self, ctx, args, ndrange, transfersize):
        self._ctx = ctx
        self._args = args
        self._ndrange = ndrange
        self._transfersize = transfersize

    def __deepcopy__(self, memo={}):
        """
        Make a deep copy of a payload.

        This means duplicating all host data, and constructing new
        OpenCL mem objects with pointers to this host data. Note that
        this DOES NOT copy the OpenCL context associated with the
        payload.
        """
        args = [clutil.KernelArg(a.string) for a in self.args]

        for newarg,arg in zip(args, self.args):
            newarg.hostdata = deepcopy(arg.hostdata, memo=memo)
            if newarg.hostdata is None:
                newarg.devdata = deepcopy(arg.devdata)
            else:
                newarg.flags = arg.flags
                newarg.devdata = cl.Buffer(self.context, newarg.flags,
                                           hostbuf=newarg.hostdata)

        return KernelPayload(self.context, args, self.ndrange, self.transfersize)

    def __eq__(self, other):
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

        for arg in self.args:
            if arg.hostdata is None or arg.is_const:
                continue

            event = cl.enqueue_copy(queue, arg.hostdata, arg.devdata,
                                    is_blocking=False)
            elapsed += get_event_time(event)

        return elapsed

    def host_to_device(self, queue):
        elapsed = 0

        for arg in self.args:
            if arg.hostdata is None:
                continue

            event = cl.enqueue_copy(queue, arg.devdata, arg.hostdata,
                                    is_blocking=False)
            elapsed += get_event_time(event)

        return elapsed

    @property
    def context(self): return self._ctx

    @property
    def args(self): return self._args

    @property
    def kargs(self): return [a.devdata for a in self._args]

    @property
    def ndrange(self): return self._ndrange

    @property
    def transfersize(self): return self._transfersize

    @staticmethod
    def _create_payload(nparray, driver, size):
        args = [clutil.KernelArg(arg.string) for arg in driver.prototype.args]
        transfer = 0

        for arg in args:
            dtype = arg.numpy_type
            arg.hostdata = None
            if arg.is_pointer:
                veclength = size * arg.vector_width
                try:
                    arg.hostdata = nparray(veclength)
                except MemoryError:
                    raise E_BAD_ARGS
                flags = cl.mem_flags.COPY_HOST_PTR
                if arg.is_const:
                    flags |= cl.mem_flags.READ_ONLY
                else:
                    flags |= cl.mem_flags.READ_WRITE
                arg.flags = flags
                arg.devdata = cl.Buffer(
                    driver.context, arg.flags, hostbuf=arg.hostdata)

                # Record transfer overhead. If it's a const buffer,
                # we're not reading back to host.
                if arg.is_const:
                    transfer += arg.hostdata.nbytes
                else:
                    transfer += 2 * arg.hostdata.nbytes
            else:
                arg.devdata = dtype(size)

        return KernelPayload(driver.context, args, (size,), transfer)

    @staticmethod
    def create_sequential(*args, **kwargs):
        return KernelPayload._create_payload(np.arange, *args, **kwargs)

    @staticmethod
    def create_random(*args, **kwargs):
        return KernelPayload._create_payload(np.random.rand, *args, **kwargs)


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
        elapsed = run_kernel_with_timeout(kernel, queue, global_size, None, *args)

        # Copy data back to host.
        for i in hostd:
            dev_buffer = args[i]
            host_buffer = hostd[i]
            event = cl.enqueue_copy(queue, host_buffer, dev_buffer)
            elapsed += get_event_time(event)
        runtimes.append(elapsed)

    mean = labmath.mean(runtimes)
    ci = labmath.confinterval(runtimes, array_mean=mean)[1] - mean
    print(fs.basename(filename), name, wgsize, transfer,
          round(mean, 6), round(ci, 6), sep=',')


def kernel(src, filename='<stdin>', devtype=cl.device_type.GPU,
           size=None, file=sys.stdout, metaout=sys.stderr, **driveopts):
    """
    Drive a kernel.
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
