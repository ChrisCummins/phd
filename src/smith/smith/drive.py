import numpy as np
import pyopencl as cl
import os


def build_program(ctx, src):
    return cl.Program(ctx, src).build()


def get_arg_prop(kernel, idx, prop, prop2str):
    name = cl.kernel_arg_info.to_string(prop)
    try:
        val = prop2str(kernel.get_arg_info(idx, prop))
    except RuntimeError as e:
        val = 'FAIL'
    return (name, val)


def get_arg_props(kernel, idx):
    props = [
        (cl.kernel_arg_info.ACCESS_QUALIFIER,
         cl.kernel_arg_access_qualifier.to_string),
        (cl.kernel_arg_info.ADDRESS_QUALIFIER,
         cl.kernel_arg_address_qualifier.to_string),
        (cl.kernel_arg_info.NAME, str),
        (cl.kernel_arg_info.TYPE_NAME, str),
        (cl.kernel_arg_info.TYPE_QUALIFIER,
         cl.kernel_arg_type_qualifier.to_string)
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


def source(src, devtype=cl.device_type.GPU, quiet=False):
    a_np = np.random.rand(50000).astype(np.float32)
    b_np = np.random.rand(50000).astype(np.float32)

    if not quiet:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = "1"

    platforms = cl.get_platforms()
    try:
        ctx = cl.Context(
            dev_type=devtype,
            properties=[(cl.context_properties.PLATFORM, platforms[0])])
    except Exception as e:
        ctx = cl.create_some_context(interactive=False)
    print("Device:", ctx.get_info(cl.context_info.DEVICES))
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

    program = build_program(ctx, src)

    for kernel in program.all_kernels():
        name = kernel.get_info(cl.kernel_info.FUNCTION_NAME)
        print("Kernel:", name)
        args = [placeholder_from_props(x) for x in args_from_kernel(kernel)]
        for arg in args:
            print("Argument:", arg)



#     kernels = program.all_kernels()
#     kernel = kernels[0]
#     get_args(kernel)

#     program.sum(queue, a_np.shape, None, a_g, b_g, res_g)

#     res_np = np.empty_like(a_np)
#     cl.enqueue_copy(queue, res_np, res_g)

#     # Check on CPU with Numpy:
#     print(res_np - (a_np + b_np))
#     print(np.linalg.norm(res_np - (a_np + b_np)))
