# jacobi-ocl

This code provides an OpenCL implementation of a Jacobi solver. The
OpenCL kernel is generated at runtime, and a variety of implementation
decisions can be made via a config file. This is primarily a research
tool for benchmarking different OpenCL-capable devices and exploring
auto-tuning and performance-portable code generation.


## Tuning implementation decisions

A JSON config file can be used to control how the OpenCL kernel should
be implemented. The following options can be tuned:

| parameter      | description                                                                                                                |
|----------------|----------------------------------------------------------------------------------------------------------------------------|
| wgsize         | A 2-element array describing the work-group size                                                                           |
| unroll         | An integer indicating how many times to unroll the main loop                                                               |
| layout         | The memory layout of the matrix (`row-major` or `col-major`)                                                               |
| conditional    | How conditional code should be generated (`branch` or `mask`)                                                              |
| fmad           | Whether `a*b+c` should use regular arithmetic operators (`op`), `fma` builtin, or `mad` builtin                            |
| divide_A       | Divide using operator (`normal`), `native_divide`, or precompute reciprocal (`precompute-global` or `precompute-constant`) |
| addrspace_b    | Which address space the `b` vector should be stored in (`global` or `constant`)                                            |
| addrspace_xold | Which address space the `xold` vector should be stored in (`global` or `constant`)                                         |
| integer        | Specifies whether integer variables should be signed or unsigned (`int` or `uint`)                                         |
| relaxed_math   | A boolean specifying whether the `-cl-fast-relaxed-math` flag should be passed to the OpenCL kernel compiler               |
| use_const      | A boolean specifying whether the `const` qualifer should be used for kernel arguments                                      |
| use_restrict   | A boolean specifying whether the `restrict` qualifer should be used for kernel arguments                                   |
| use_mad24      | A boolean specifying whether integer `a*b+c` operations should use the `mad24` builtin                                     |
| const_norder   | A boolean specifying whether the `norder` parameter should be compiled into the kernel as a constant value                 |
| const_wgsize   | A boolean specifying whether the work-group size should be compiled into the kernel as a constant value                    |
| coalesce_cols  | A boolean specifying whether adjacent work-items should access memory in a coalesced or strided manner                     |

An example JSON configuration file:

    {
      "wgsize": [
        128,
        1
      ],
      "unroll": 8,
      "layout": "row-major",
      "conditional": "mask",
      "fmad": "fma",
      "divide_A": "precompute-constant",
      "addrspace_b": "constant",
      "addrspace_xold": "constant",
      "integer": "int",
      "relaxed_math": true,
      "use_const": true,
      "use_restrict": true,
      "use_mad24": true,
      "const_wgsize": false,
      "const_norder": false,
      "coalesce_cols": true
    }


## Command-line options

| option                      | description                         |
|-----------------------------|-------------------------------------|
| -n, --norder                | matrix order                        |
| -i, --iterations            | number of iterations to run         |
| -f, --datatype              | data type (float or double)         |
| -c, --config                | config file to use                  |
| -k, --convergence-frequency | how often to check for convergence  |
| -t, --convergence-tolerance | error tolerance for stopping solver |
| -p, --print-kernel          | print out the generated kernel      |
| -l, --list                  | list available OpenCL devices       |
| -d, --device                | select OpenCL device by index       |

By default, convergence checking is disabled. If enabled, the solver
will until the error reaches the tolerance specified.


## Example output

    --------------------------------
    MATRIX     = 2048x2048
    ITERATIONS = 10000
    DATATYPE   = double
    Convergence checking disabled
    --------------------------------
    Work-group size    = [128, 1]
    Unroll factor      = 8
    Data layout        = row-major
    Conditional        = mask
    fmad               = fma
    Divide by A        = precompute-constant
    b address space    = constant
    xold address space = constant
    Integer type       = int
    Relaxed math       = True
    Use restrict       = True
    Use const pointers = True
    Use mad24          = True
    Constant norder    = False
    Constant wgsize    = False
    Coalesce columns   = True
    --------------------------------
    Using 'Tesla K20m'
    Runtime = 2.41s (10000 iterations)
    Error   = 0.001228
