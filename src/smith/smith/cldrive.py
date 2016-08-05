from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from copy import deepcopy
from functools import partial,wraps
from random import randrange
from threading import Thread
from io import StringIO
from io import open

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


class CLDriveException(smith.SmithException): pass
class OpenCLDriverException(CLDriveException): pass
class OpenCLNotSupported(OpenCLDriverException): pass

class KernelDriverException(CLDriveException): pass

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
        # Safety first, kids:
        assert(type(ctx) == cl.Context)
        assert(type(source) == str or type(source) == unicode)

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
    def __call__(self, queue, payload):
        # Safety first, kids:
        assert(type(queue) == cl.CommandQueue)
        assert(type(payload) == KernelPayload)

        # First off, let's clear any existing tasks in the command
        # queue:
        queue.flush()

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

        # Check that everything is done before we finish:
        queue.flush()

        return output

    def __repr__(self):
        return self.source

    def validate(self, queue, size=16):
        assert(type(queue) == cl.CommandQueue)

        def assert_constraint(constraint, err=CLDriveException):
            if not constraint:
                raise err

        # Create payloads.
        A1in = KernelPayload.create_sequential(self, size)
        A2in = deepcopy(A1in)

        B1in = KernelPayload.create_random(self, size)
        B2in = deepcopy(B1in)

        # Input constraints.
        assert_constraint(A1in == A2in, E_BAD_DRIVER)
        assert_constraint(B1in == B2in, E_BAD_DRIVER)
        assert_constraint(A1in != B1in, E_BAD_DRIVER)

        k = partial(self, queue)
        A1out = k(A1in)
        B1out = k(B1in)
        A2out = k(A2in)
        B2out = k(B2in)

        # outputs must be consistent across runs:
        assert_constraint(A1out == A2out, E_NONDETERMINISTIC)
        assert_constraint(B1out == B2out, E_NONDETERMINISTIC)

        # outputs must depend on inputs:
        if any(not x.is_const for x in self.prototype.args):
            assert_constraint(A1out != B1out, E_INPUT_INSENSITIVE)

        # outputs must be different from inputs:
        assert_constraint(A1in != A1out, E_NO_OUTPUTS)
        assert_constraint(B1in != B1out, E_NO_OUTPUTS)

    def profile(self, queue, size=16, must_validate=False,
                out=sys.stdout, metaout=sys.stderr):
        assert(type(queue) == cl.CommandQueue)

        try:
            self.validate(queue, size)
        except CLDriveException as e:
            print(self.name, type(e).__name__, sep=',', file=metaout)
            if must_validate:
                return

        P = KernelPayload.create_random(self, size)
        k = partial(self, queue)

        while len(self.runtimes) < 10:
            k(P)

        wgsize = round(labmath.mean(self.wgsizes))
        transfer = round(labmath.mean(self.transfers))
        mean = labmath.mean(self.runtimes)
        ci = labmath.confinterval(self.runtimes, array_mean=mean)[1] - mean
        print(unicode(self.name), unicode(wgsize), unicode(transfer), unicode(round(mean, 6)), unicode(round(ci, 6)),
              sep=unicode(','), file=out)

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
        except Exception as e:
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
            if arg.hostdata is None:
                newarg.hostdata = None
                newarg.devdata = deepcopy(arg.devdata)
            else:
                newarg.hostdata = deepcopy(arg.hostdata, memo=memo)
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
                if len(x.hostdata) != len(y.hostdata):
                    return False
                if any(e1 != e2 for e1,e2 in zip(x.hostdata,y.hostdata)):
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
    def kargs(self): return [a.devdata for a in self.args]

    @property
    def ndrange(self): return self._ndrange

    @property
    def transfersize(self): return self._transfersize

    def __repr__(self):
        return ('\n'.join([repr(x.hostdata) for x in self.args]))

    @staticmethod
    def _create_payload(nparray, driver, size):
        args = [clutil.KernelArg(arg.string) for arg in driver.prototype.args]
        transfer = 0

        for arg in args:
            dtype = arg.numpy_type
            arg.hostdata = None
            if arg.is_pointer:
                veclength = size * arg.vector_width

                # Allocate host memory and populate with values:
                try:
                    arg.hostdata = nparray(veclength)
                except MemoryError as e:
                    raise E_BAD_ARGS(e)

                # Determine flags to pass to OpenCL buffer creation:
                flags = cl.mem_flags.COPY_HOST_PTR
                if arg.is_const:
                    flags |= cl.mem_flags.READ_ONLY
                else:
                    flags |= cl.mem_flags.READ_WRITE
                arg.flags = flags

                # Allocate device memory:
                try:
                    arg.devdata = cl.Buffer(
                        driver.context, arg.flags, hostbuf=arg.hostdata)
                except Exception as e:
                    raise E_BAD_ARGS(e)

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


def kernel(src, filename='<stdin>', devtype=cl.device_type.GPU,
           size=None, must_validate=False):
    """
    Drive a kernel.
    """
    ctx, queue = init_opencl(devtype=devtype)
    driver = KernelDriver(ctx, src)

    # If no size is given, pick one.
    if size is None:
        size = 2 ** randrange(4, 15)

    out = StringIO()
    metaout = StringIO()
    driver.profile(queue, size=size, must_validate=must_validate,
                   out=out, metaout=metaout)

    stdout = out.getvalue()
    stderr = metaout.getvalue()

    # Print results:
    [print(filename, x, sep=',')
     for x in stdout.split('\n') if x]
    [print(filename, x, sep=',', file=sys.stderr)
     for x in stderr.split('\n') if x]


def file(path, **kwargs):
    with open(fs.path(path)) as infile:
        src = infile.read()
        for kernelsrc in clutil.get_cl_kernels(src):
            kernel(kernelsrc, filename=fs.path(path), **kwargs)


def directory(path, **kwargs):
    files = fs.ls(fs.path(path), abspaths=True, recursive=True)
    for path in [f for f in files if f.endswith('.cl')]:
        file(path, **kwargs)
