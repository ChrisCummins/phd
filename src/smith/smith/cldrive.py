from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

from copy import deepcopy
from functools import partial,wraps
from io import open
from random import randrange
from subprocess import check_output
from threading import Thread

import numpy as np
import os
import pyopencl as cl
import signal
import sys

import labm8
from labm8 import fs
from labm8 import math as labmath

import smith
from smith import clutil
from smith import config as cfg

# Python 2 and Python 3 have different StringIO classes.
# See: http://stackoverflow.com/a/19243243
if labm8.is_python3():
    from io import StringIO
else:
    from StringIO import StringIO


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


def hang_requires_restart():
    """
    Does an OpenCL kernel hang require a system restart?
    """
    # FIXME: return gethostname() == "monza"
    return True



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
        except Exception as e:
            raise E_BAD_ARGS(e)

        # Execute kernel
        local_size_x = min(output.ndrange[0], 128)
        event = self.kernel(queue, output.ndrange, (local_size_x,), *kargs)
        elapsed += get_event_time(event)

        # Copy data back to host and get time.
        elapsed += output.device_to_host(queue)

        # Record workgroup size.
        self.wgsizes.append(local_size_x)

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
        """
        Output format (CSV):

            out:      <kernel> <wgsize> <transfer> <runtime> <ci>
            metaout:  <error> <kernel>
        """
        assert(type(queue) == cl.CommandQueue)

        try:
            self.validate(queue, size)
        except CLDriveException as e:
            print(type(e).__name__, self.name, sep=',', file=metaout)
            if must_validate:
                return

        P = KernelPayload.create_random(self, size)
        k = partial(self, queue)

        while len(self.runtimes) < 10:
            k(P)

        wgsize = int(round(labmath.mean(self.wgsizes)))
        transfer = int(round(labmath.mean(self.transfers)))
        mean = labmath.mean(self.runtimes)
        ci = labmath.confinterval(self.runtimes, array_mean=mean)[1] - mean
        print(self.name, wgsize, transfer, round(mean, 6), round(ci, 6),
              sep=',', file=out)

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
            if arg.hostdata is None and arg.is_local:
                # Copy a local memory buffer.
                newarg.hostdata = None
                newarg.bufsize = arg.bufsize
                newarg.devdata = cl.LocalMemory(newarg.bufsize)
            elif arg.hostdata is None:
                # Copy a scalar value.
                newarg.hostdata = None
                newarg.devdata = deepcopy(arg.devdata)
            else:
                # Copy a global memory buffer.
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

        try:
            for arg in args:
                arg.hostdata = None

                dtype = arg.numpy_type
                veclength = size * arg.vector_width

                if arg.is_pointer and arg.is_local:
                    # If arg is a pointer to local memory, then we
                    # create a read/write buffer:
                    nonbuf = nparray(veclength)
                    arg.bufsize = nonbuf.nbytes
                    arg.devdata = cl.LocalMemory(arg.bufsize)
                elif arg.is_pointer:
                    # If arg is a pointer to global memory, then we
                    # allocate host memory and populate with values:
                    arg.hostdata = nparray(veclength).astype(dtype)

                    # Determine flags to pass to OpenCL buffer creation:
                    arg.flags = cl.mem_flags.COPY_HOST_PTR
                    if arg.is_const:
                        arg.flags |= cl.mem_flags.READ_ONLY
                    else:
                        arg.flags |= cl.mem_flags.READ_WRITE

                    # Allocate device memory:
                    arg.devdata = cl.Buffer(
                        driver.context, arg.flags, hostbuf=arg.hostdata)

                    # Record transfer overhead. If it's a const buffer,
                    # we're not reading back to host.
                    if arg.is_const:
                        transfer += arg.hostdata.nbytes
                    else:
                        transfer += 2 * arg.hostdata.nbytes
                else:
                    # If arg is not a pointer, then it's a scalar value:
                    arg.devdata = dtype(size)
        except Exception as e:
            raise E_BAD_ARGS(e)

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

    Output format (CSV):

        out:      <file> <size> <kernel> <wgsize> <transfer> <runtime> <ci>
        metaout:  <file> <size> <error> <kernel>
    """
    try:
        ctx, queue = init_opencl(devtype=devtype)
        driver = KernelDriver(ctx, src)
    except Exception as e:
        print(filename, size, type(e).__name__, '-', sep=',', file=sys.stderr)
        return

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
    [print(filename, size, line, sep=',')
     for line in stdout.split('\n') if line]
    [print(filename, size, line, sep=',', file=sys.stderr)
     for line in stderr.split('\n') if line]


def file(path, **kwargs):
    with open(fs.path(path)) as infile:
        src = infile.read()
        for kernelsrc in clutil.get_cl_kernels(src):
            kernel(kernelsrc, filename=fs.basename(path), **kwargs)


def directory(path, **kwargs):
    files = fs.ls(fs.path(path), abspaths=True, recursive=True)
    for path in [f for f in files if f.endswith('.cl')]:
        file(path, **kwargs)
