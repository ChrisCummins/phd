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
import sys

import numpy as np
import pyopencl as cl

from cldrive import *


class NDRange(namedtuple('NDRange', ['x', 'y', 'z'])):
    __slots__ = ()

    def __repr__(self):
        return f"[self.x, self.y, self.z]"

    @property
    def product(self):
        return self.x * self.y * self.z


def drive(env: OpenCLEnvironment, src: str, inputs: np.array,
          gsize: NDRange, lsize: NDRange, timeout: float=-1,
          optimizations: bool=True, profiling: bool=False, debug: bool=False):
    """
    OpenCL kernel.

    Arguments:
        env (OpenCLEnvironment): The OpenCL environment to run the
            kernel in.
        src (str): The OpenCL kernel source.
        optimizations (bool, optional): Whether to enable or disbale OpenCL
            compiler optimizations.
        profiling (bool, optional): If true, record OpenCLevent times for
        timeout (float, optional): Cancel execution if it has not completed
            after this many seconds. A value <= 0 means never time out.
        debug (bool, optional): If true, silence the OpenCL compiler.
        data transfers and kernel executions.

    Returns:
        np.array: A numpy array of the same shape as the inputs, with the
            values after running the OpenCL kernel.

    Raises:
        ValueError: if input types are incorrect.
        TypeError: if an input is of an incorrect type.
        LogicError: if the input types do not match OpenCL kernel types.
        RuntimeError: if OpenCL program fails to build or run.

    TODO:
        * Implement profiling
    """
    def log(*args, **kwargs):
        if debug:
            print(*args, **kwargs, file=sys.stderr)

    # assert input types
    assert_or_raise(isinstance(env, OpenCLEnvironment), ValueError,
                    "env argument is of incorrect type")
    assert_or_raise(isinstance(src, str), ValueError,
                    "source is not a string")
    assert_or_raise(len(gsize) == 3, TypeError)
    assert_or_raise(len(lsize) == 3, TypeError)
    gsize, lsize = NDRange(*gsize), NDRange(*lsize)

    # OpenCL compiler flags
    if optimizations:
        build_flags = []
        log("OpenCL optimizations: on")
    else:
        build_flags = ['-cl-opt-disable']
        log("OpenCL optimizations: off")

    if debug:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    else:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

    # parse args first as this is most likely to raise an error
    args = extract_args(src)

    ctx, queue = env.ctx_queue

    try:
        program = cl.Program(ctx, src).build(build_flags)
    except cl.RuntimeError as e:
        raise RuntimeError from e

    kernels = program.all_kernels()
    # extract_args() should already have raised an error if there's more
    # than one kernel:
    assert(len(kernels) == 1)
    kernel = kernels[0]

    # copy inputs into the expected data types
    data = np.array([np.array(d).astype(a.numpy_type)
                     for d, a in zip(inputs, args)])

    # sanity check that there are enough the correct number of inputs
    data_indices = [i for i, arg in enumerate(args) if not arg.is_local]
    assert_or_raise(len(data_indices) == len(data), ValueError,
                    "Incorrect number of inputs provided")

    # scalar_gsize is the product of the global NDRange.
    scalar_gsize, scalar_lsize = gsize.product, lsize.product

    assert_or_raise(gsize.product >= 1, ValueError,
                    f"Scalar global size {gsize.product} must be >= 1")
    assert_or_raise(lsize.product >= 1, ValueError,
                    f"Scalar local size {lsize.product} must be >= 1")
    assert_or_raise(gsize > lsize, ValueError,
                    f"Global size {gsize} must be larger than local size {lsize}")

    log(f"""\
3-D global size {gsize.product} = {gsize}
3-D local size {lsize.product} = {lsize}""")

    # buffer size is the scalar global size, or the size of the largest
    # input, which is bigger
    buf_size = max(gsize.product, *[x.size for x in data])

    # assemble argtuples
    ArgTuple = namedtuple('ArgTuple', ['hostdata', 'devdata'])
    argtuples = []
    data_i = 0
    for i, arg in enumerate(args):
        if arg.is_global:
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
        elif arg.is_local:
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
                    "failed to set input arguments")

    # clear any existing tasks in the command queue:
    queue.flush()

    # copy host -> device
    for argtuple in argtuples:
        if argtuple.hostdata is not None:
            cl.enqueue_copy(queue, argtuple.devdata, argtuple.hostdata,
                            is_blocking=False)

    kernel_args = [argtuple.devdata for argtuple in argtuples]

    try:
        kernel.set_args(*kernel_args)
    except cl.LogicError as e:
        raise TypeError(e)

    # run the kernel
    kernel(queue, gsize, lsize, *kernel_args)

    # copy device -> host
    for arg, argtuple in zip(args, argtuples):
        if argtuple.hostdata is not None and not arg.is_const:
            cl.enqueue_copy(
                queue, argtuple.hostdata, argtuple.devdata,
                is_blocking=False)

    # wait for queue to finish
    queue.flush()

    return data
