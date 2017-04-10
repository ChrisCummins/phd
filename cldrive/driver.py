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
import pickle
import re
import sys

from contextlib import suppress
from pkg_resources import resource_filename
from signal import Signals
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from typing import Union

import numpy as np

from cldrive import *


class Timeout(RuntimeError):
    """ thrown if kernel executions fails to complete before timeout """
    def __init__(self, timeout: int):
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"execution failed to complete with {self.timeout} seconds"


class PorcelainError(RuntimeError):
    """ raised if porcelain subprocess exits with non-zero return code """
    def __init__(self, status: Union[int, str]):
        self.status = status

    def __repr__(self) -> str:
        return f"porcelain subprocess exited with return code {self.status}"


class NDRange(namedtuple('NDRange', ['x', 'y', 'z'])):
    """
    A 3 dimensional NDRange tuple. Has components x,y,z.

    Attributes
    ----------
    x : int
        x component.
    y : int
        y component.
    z : int
        z component.

    Examples
    --------
    >>> NDRange(1, 2, 3)
    [1, 2, 3]

    >>> NDRange(1, 2, 3).product
    6

    >>> NDRange(10, 10, 10) > NDRange(10, 9, 10)
    True
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return f"[{self.x}, {self.y}, {self.z}]"

    @property
    def product(self) -> int:
        """ linear product is x * y * z """
        return self.x * self.y * self.z

    def __eq__(self, rhs: 'NDRange') -> bool:
        return (self.x == rhs.x and self.y == rhs.y and self.z == rhs.z)

    def __gt__(self, rhs: 'NDRange') -> bool:
        return (self.product > rhs.product and
                self.x >= rhs.x and self.y >= rhs.y and self.z >= rhs.z)

    def __ge__(self, rhs: 'NDRange') -> bool:
        return self == rhs or self > rhs

    @staticmethod
    def from_str(string: str) -> 'NDRange':
        """
        Parse an NDRange from a string of format 'x,y,z'.

        Parameters
        ----------
        string : str
            Comma separated NDRange values.

        Returns
        -------
        NDRange
            Parsed NDRange.

        Raises
        ------
        ValueError:
            If the string does not contain three comma separated integers.

        Examples
        --------
        >>> NDRange.from_str('10,11,3')
        [10, 11, 3]

        >>> NDRange.from_str('10,11,3,1')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError
        """
        components = string.split(',')
        if not len(components) == 3:
            raise ValueError(f"invalid NDRange '{string}'")

        x, y, z = int(components[0]), int(components[1]), int(components[2])

        return NDRange(x, y, z)


def drive(env: OpenCLEnvironment, src: str, inputs: np.array,
          gsize: NDRange, lsize: NDRange, timeout: int=-1,
          optimizations: bool=True, profiling: bool=False,
          debug: bool=False) -> np.array:
    """
    Drive an OpenCL kernel.

    Executes an OpenCL kernel on the given environment, over the given inputs.
    Execution is performed in a subprocess.

    Parameters
    ----------
    env : OpenCLEnvironment
        The OpenCL environment to run the kernel in.
    src : str
        The OpenCL kernel source.
    inputs : np.array
        The input data to the kernel.
    optimizations : bool, optional
        Whether to enable or disbale OpenCL compiler optimizations.
    profiling : bool, optional
        If true, print OpenCLevent times for data transfers and kernel
        executions to stderr.
    timeout : int, optional
        Cancel execution if it has not completed after this many seconds.
        A value <= 0 means never time out.
    debug : bool, optional
        If true, silence the OpenCL compiler.

    Returns
    -------
    np.array
        A numpy array of the same shape as the inputs, with the values after
        running the OpenCL kernel.

    Raises
    ------
    ValueError
        If input types are incorrect.
    TypeError
        If an input is of an incorrect type.
    LogicError
        If the input types do not match OpenCL kernel types.
    PorcelainError
        If the OpenCL subprocess exits with non-zero return  code.
    RuntimeError
        If OpenCL program fails to build or run.

    Examples
    --------
    A simple kernel which doubles its inputs:

    >>> src = "kernel void A(global int* a) { a[get_global_id(0)] *= 2; }"
    >>> inputs = [[1, 2, 3, 4, 5]]
    >>> drive(make_env(), src, inputs, gsize=(5,1,1), lsize=(1,1,1))
    array([[ 2,  4,  6,  8, 10]], dtype=int32)
    """
    def log(*args, **kwargs):
        if debug:
            print("[cldrive] ", end="", file=sys.stderr)
            print(*args, **kwargs, file=sys.stderr)

    # assert input types
    assert_or_raise(isinstance(env, OpenCLEnvironment), ValueError,
                    "env argument is of incorrect type")
    assert_or_raise(isinstance(src, str), ValueError,
                    "source is not a string")

    # validate global and local sizes
    assert_or_raise(len(gsize) == 3, TypeError)
    assert_or_raise(len(lsize) == 3, TypeError)
    gsize, lsize = NDRange(*gsize), NDRange(*lsize)

    assert_or_raise(gsize.product >= 1, ValueError,
                    f"Scalar global size {gsize.product} must be >= 1")
    assert_or_raise(lsize.product >= 1, ValueError,
                    f"Scalar local size {lsize.product} must be >= 1")
    assert_or_raise(gsize >= lsize, ValueError,
                    f"Global size {gsize} must be larger than local size {lsize}")

    # parse args in this process since we want to preserve the sueful exception
    # type
    args = extract_args(src)

    # check that the number of inputs is correct
    args_with_inputs = [i for i, arg in enumerate(args)
                        if not arg.address_space == 'local']
    assert_or_raise(len(args_with_inputs) == len(inputs), ValueError,
                    "Kernel expects {} inputs, but {} were provided".format(
                        len(args_with_inputs), len(inputs)))

    # all inputs must have some length
    for i, x in enumerate(inputs):
        assert_or_raise(len(x), ValueError, f"Input {i} has size zero")

    # copy inputs into the expected data types
    data = np.array([np.array(d).astype(a.numpy_type)
                     for d, a in zip(inputs, args)])

    job = {
        "env": env,
        "src": src,
        "args": args,
        "data": data,
        "gsize": gsize,
        "lsize": lsize,
        "optimizations": optimizations,
        "profiling": profiling
    }

    with NamedTemporaryFile('rb+', prefix='cldrive-', suffix='.job') as tmpfile:
        porcelain_job_file = tmpfile.name

        # write job file
        pickle.dump(job, tmpfile)
        tmpfile.flush()

        # enforce timeout using sigkill
        if timeout > 0:
            cli = ["timeout", "--signal=9", str(int(timeout))]
        else:
            cli = []
        cli += [sys.executable, __file__, porcelain_job_file]

        cli_str = " ".join(cli)
        log("Porcelain invocation:", cli_str)

        # fork and run
        process = Popen(cli, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        status = process.returncode

        if debug:
            print(stdout.decode('utf-8').strip(), file=sys.stderr)
            print(stderr.decode('utf-8').strip(), file=sys.stderr)
        elif profiling:
            # print profiling output when not in debug mode:
            for line in stderr.decode('utf-8').split('\n'):
                if re.match(r'\[cldrive\] .+ time: [0-9]+\.[0-9]+ ms', line):
                    print(line, file=sys.stderr)
        log(f"Porcelain return code: {status}")

        # test for non-zero exit codes. The porcelain subprocess catches
        # exceptions and completes gracefully, so a non-zero return code is
        # indicative of a more serious problem.
        #
        # FIXME: I'm seeing a number of SIGABRT returncodes which I can't
        # explain. However, ignoring them seems to not cause a problem ...
        if status != 0 and status != -Signals['SIGABRT'].value:
            # a negative return code means a signal. Try and convert the value
            # into a signal name.
            with suppress(ValueError):
                status = Signals(-status).name

            if status == "SIGKILL":
                raise Timeout(timeout)
            else:
                raise PorcelainError(status)

        # read result
        tmpfile.seek(0)
        rets = pickle.load(tmpfile)

        outputs = rets["outputs"]
        err = rets["err"]

        if err:  # porcelain raised an exception, re-raise it
            raise err
        else:
            return outputs


def __porcelain_exec(path: str) -> np.array:
    """ here be dragons """
    import pyopencl as cl  # defered loading of OpenCL library

    def log(*args, **kwargs):
        print("[cldrive] ", end="", file=sys.stderr)
        print(*args, **kwargs, file=sys.stderr)

    def get_event_time(event: cl.Event) -> float:
        """
        Block until OpenCL event has completed and return time delta
        between event submission and end, in milliseconds.

        Arguments:
            event (cl.Event): Event handle.

        Returns:
            float: Elapsed time, in milliseconds.
        """
        event.wait()
        tstart = event.get_profiling_info(cl.profiling_info.START)
        tend = event.get_profiling_info(cl.profiling_info.END)
        return (tend - tstart) / 1000000

    # restore job
    with open(path, 'rb') as infile:
        job = pickle.load(infile)

    env = job["env"]
    src = job["src"]
    args = job["args"]
    data = job["data"]
    gsize = job["gsize"]
    lsize = job["lsize"]
    optimizations = job["optimizations"]
    profiling = job["profiling"]

    ctx, queue = env.ctx_queue(profiling=profiling)

    # CLSmith cl_launcher compatible logging output. See:
    #    https://github.com/ChrisCummins/CLSmith/blob/master/src/CLSmith/cl_launcher.c
    device = queue.get_info(cl.command_queue_info.DEVICE)
    device_name = device.get_info(cl.device_info.NAME)

    platform = device.get_info(cl.device_info.PLATFORM)
    platform_name = platform.get_info(cl.platform_info.NAME)

    log(f"Platform: {platform_name}")
    log(f"Device: {device_name}")

    log(f"3-D global size {gsize.product} = {gsize}")
    log(f"3-D local size {lsize.product} = {lsize}")

    # Additional logging output for inputs:
    log("Number of kernel arguments:", len(args))
    log("Kernel arguments:", ", ".join(str(a) for a in args))
    log("Kernel input sizes: [", ", ".join(str(x.size) for x in data), "]",
        sep="")

    # buffer size is the scalar global size, or the size of the largest
    # input, whichever is bigger
    if len(data):
        buf_size = max(gsize.product, *[x.size for x in data])
    else:
        buf_size = gsize.product

    # parent process determines whether or not to silence this output
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    # always compile kernels, don't use cached binaries
    os.environ['PYOPENCL_NO_CACHE'] = '1'

    # OpenCL compiler flags
    if optimizations:
        build_flags = []
        log("OpenCL optimizations: on")
    else:
        build_flags = ['-cl-opt-disable']
        log("OpenCL optimizations: off")

    if profiling:
        log("OpenCL profiling: on")
    else:
        log("OpenCL profiling: off")

    try:
        program = cl.Program(ctx, src).build(build_flags)
        log("Compilation succeeded")
    except cl.RuntimeError as e:
        raise RuntimeError from e

    kernels = program.all_kernels()
    # extract_args() should already have raised an error if there's more
    # than one kernel:
    assert(len(kernels) == 1)
    kernel = kernels[0]

    # clear any existing tasks in the command queue
    queue.flush()
    log("Command queue flushed")

    # assemble argtuples
    ArgTuple = namedtuple('ArgTuple', ['hostdata', 'devdata'])
    argtuples = []
    data_i = 0
    for i, arg in enumerate(args):
        if arg.address_space == 'global' or arg.address_space == 'constant':
            data[data_i] = data[data_i].astype(arg.numpy_type)
            hostdata = data[data_i]
            # determine flags to pass to OpenCL buffer creation:
            flags = cl.mem_flags.COPY_HOST_PTR
            if arg.is_const:
                flags |= cl.mem_flags.READ_ONLY
            else:
                flags |= cl.mem_flags.READ_WRITE
            buf = cl.Buffer(ctx, flags, hostbuf=hostdata)

            devdata, data_i = buf, data_i + 1
        elif arg.address_space == 'local':
            nbytes = buf_size * arg.vector_width * arg.numpy_type.itemsize
            buf = cl.LocalMemory(nbytes)

            hostdata, devdata = None, buf
        elif not arg.is_pointer:
            hostdata = None
            devdata, data_i = arg.numpy_type(data[data_i]), data_i + 1
        else:
            # argument is neither global or local, but is a pointer?
            raise ValueError(f"unknown argument type '{arg}'")
        argtuples.append(ArgTuple(hostdata=hostdata, devdata=devdata))

    assert_or_raise(len(data) == data_i, ValueError,
                    "failed to interpret inputs")
    log("Device memory allocated")

    kernel_args = [argtuple.devdata for argtuple in argtuples]

    try:
        kernel.set_args(*kernel_args)
        log("Set kernel arguments")
    except cl.LogicError as e:
        raise ValueError(f"failed to set kernel args") from e

    upload_elapsed = 0
    if len(argtuples):
        for argtuple in argtuples:
            if argtuple.hostdata is not None:
                event = cl.enqueue_copy(queue, argtuple.devdata, argtuple.hostdata,
                                        is_blocking=False)
                if profiling:
                    upload_elapsed += get_event_time(event)

        if profiling:
            log(f"Host -> Device transfers time: {upload_elapsed:.6f} ms")
        else:
            log("Host -> Device transfers enqueued")

    # run the kernel
    event = kernel(queue, gsize, lsize, *kernel_args)
    log("Kernel execution enqueued")

    if profiling:
        runtime = get_event_time(event)
        log(f"Kernel execution time: {runtime:.6f} ms")


    download_elapsed = 0
    if len(argtuples):
        for arg, argtuple in zip(args, argtuples):
            # const arguments are unmodified
            if argtuple.hostdata is not None and not arg.is_const:
                cl.enqueue_copy(queue, argtuple.hostdata, argtuple.devdata,
                                is_blocking=False)
                if profiling:
                    download_elapsed += get_event_time(event)

        if profiling:
            log(f"Device -> Host transfers time: {download_elapsed:.6f} ms")
        else:
            log("Device -> Host transfers enqueued")



    # wait for OpenCL commands to complete
    queue.flush()
    log("Command queue flushed")

    return data


# entry point for porcelain incvocation
if __name__ == "__main__":
    path = sys.argv[1]

    outputs = None
    err = None

    try:
        outputs = __porcelain_exec(path)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        err = e

    with open(path, 'wb') as outfile:
        pickle.dump({
            "outputs": outputs,
            "err": err
        }, outfile)

    print("[cldrive] Porcelain subprocess complete", file=sys.stderr)
    sys.exit(0)
