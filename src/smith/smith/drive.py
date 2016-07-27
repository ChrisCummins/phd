import numpy as np
import pyopencl as cl
import os

def build_program(ctx, src):
    return cl.Program(ctx, src).build()


def get_arg_prop(kernel, idx, prop):
    name = cl.kernel_arg_info.to_string(prop)
    try:
        val = kernel.get_arg_info(idx, prop)
    except RuntimeError as e:
        val = 'FAIL'
    print('    {}: {}'.format(name, val))
    return (name, val)


def get_arg(kernel, idx):
    print("ARG", idx)
    props = [
        cl.kernel_arg_info.ACCESS_QUALIFIER,
        cl.kernel_arg_info.ADDRESS_QUALIFIER,
        cl.kernel_arg_info.NAME,
        cl.kernel_arg_info.TYPE_NAME,
        cl.kernel_arg_info.TYPE_QUALIFIER
    ]
    return dict(get_arg_prop(kernel, idx, x) for x in props)


def get_args(kernel):
    nargs = kernel.get_info(cl.kernel_info.NUM_ARGS)
    return [get_arg(kernel, i) for i in range(nargs)]


def source(src, devtype=cl.device_type.GPU, quiet=False):
    print("Hello, world!")
    a_np = np.random.rand(50000).astype(np.float32)
    b_np = np.random.rand(50000).astype(np.float32)

    if not quiet:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = "1"

    ctx = cl.Context(dev_type=devtype)
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

    program = build_program(ctx, src)

    for kernel in program.all_kernels():
        get_args(kernel)

    # program = build_program(ctx, """
# __kernel void sum(
#     __global const float *a, __global const float *b_g, __global float *res_g)
# {
#   int gid = get_global_id(0);
#   res_g[gid] = a[gid] + b_g[gid];
# }
# """)

#     kernels = program.all_kernels()
#     kernel = kernels[0]
#     get_args(kernel)

#     program.sum(queue, a_np.shape, None, a_g, b_g, res_g)

#     res_np = np.empty_like(a_np)
#     cl.enqueue_copy(queue, res_np, res_g)

#     # Check on CPU with Numpy:
#     print(res_np - (a_np + b_np))
#     print(np.linalg.norm(res_np - (a_np + b_np)))
