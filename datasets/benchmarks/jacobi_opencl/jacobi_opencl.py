# Copyright (c) 2015-16, James Price and Simon McIntosh-Smith,
# University of Bristol. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import json
import math
import time

import argparse
import collections
import numpy
import pyopencl as CL
import signal

from labm8 import app

JacobiBenchmarkRun = collections.namedtuple(
    'JacobiBenchmarkRun', ['runtime', 'error', 'iteration_count'])

h_A = None
h_b = None


class timeout:

  def __init__(self, seconds=1):
    self.seconds = seconds

  def handle_timeout(self, signum, frame):
    raise Exception('timeout')

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.handle_timeout)
    signal.alarm(self.seconds)

  def __exit__(self, type, value, traceback):
    signal.alarm(0)


class Tuner:

  def __init__(self, config, norder, datatype, context, max_error, max_runtime):
    self.norder = norder
    self.datatype = datatype
    self.context = context
    self.max_error = max_error
    self.max_runtime = max_runtime

    device = context.devices[0]
    self.max_wgsize = device.max_work_group_size

    config['use_wgsize_x'] = config['wgsize'][0] > 1
    config['use_wgsize_y'] = config['wgsize'][1] > 1
    self.config = config

    self.best = None
    self.results = dict()

  # Evaluate a configuration and return the runtime
  def evaluate(self, wgsize_config, iterations):
    self.config['wgsize'] = wgsize_config[:]
    try:
      if wgsize_config in self.results:
        result = self.results[wgsize_config]
        if result:
          print('%-16s : %.4gs [cached]' % (wgsize_config, result))
        else:
          print('%-16s : failed [cached]' % (wgsize_config,))
        return result

      max_runtime = self.max_runtime
      if self.best:
        max_runtime = int(self.best[1]) + 2

      result = RunJacobiBenchmarkConfig(self.config, self.norder, iterations,
                                        self.datatype, self.context,
                                        max_runtime, 0, 0)

      if self.max_error > 0 and not result[1] < self.max_error:
        raise Exception('verification failed')

      print('%-16s : %.4gs' % (wgsize_config, result[0]))

      if not self.best or result[0] < self.best[1]:
        self.best = (wgsize_config, result[0], result[1], result[2])

      self.results[wgsize_config] = result[0]
      return result[0]
    except Exception as e:
      print('%-16s : %s' % (wgsize_config, str(e)))
      self.results[wgsize_config] = None
      return None

  # Run a steepest ascent hill climber starting at seed
  def local_search(self, seed, iterations):
    print('Performing local search starting at %s' % (seed,))

    current = seed
    current_runtime = self.evaluate(seed, iterations)

    itr = 0
    tuning = True
    while tuning:
      # Generate neighbour list
      neighbours = []

      if self.config['use_wgsize_x']:
        tc = (current[0] / 2, current[1])
        if self.valid(tc): neighbours.append(tc)
        tc = (current[0] * 2, current[1])
        if self.valid(tc): neighbours.append(tc)
      if self.config['use_wgsize_y']:
        tc = (current[0], current[1] / 2)
        if self.valid(tc): neighbours.append(tc)
        tc = (current[0], current[1] * 2)
        if self.valid(tc): neighbours.append(tc)

      tuning = False

      # Evaluate neighbours
      for cfg in neighbours:
        runtime = self.evaluate(cfg, iterations)
        if not runtime:
          continue
        if not current_runtime or runtime < current_runtime:
          # Move to improved neighbour
          current = cfg
          current_runtime = runtime
          tuning = True

      if current_runtime:
        print('Iteration %d: %.4gs %s' % (itr, current_runtime, current))
      else:
        print('Iteration %d: -' % itr)
      itr += 1

  # Reset the record of results tried and best so far
  def reset(self):
    self.best = None
    self.results = dict()

  # Select k evenly spaced elements from l
  def select_uniform(self, l, k):
    n = len(l)
    indices = (int(round(i * (n / float(k)))) for i in range(k))
    return [l[i] for i in indices]

  # Check if a wgsize configuration is sensible for the target device
  def sensible(self, wgsize_config):
    device = self.context.devices[0]
    wgsize = wgsize_config[0] * wgsize_config[1]
    groups = (self.norder * self.norder) / wgsize

    if device.type == CL.device_type.GPU:
      # Ensure work-groups are not too small
      if wgsize < 16:
        return False

    # Make sure there are enough groups to fill at least half of the device
    if groups < device.max_compute_units / 2:
      return False

    return True

  # Evaluate num_tests uniformly generated wgsize configurations
  def uniform_search(self, num_tests, iterations):
    print('Performing uniform search with %d configurations' % num_tests)

    # Generate list of all valid and sensible wgsize configurations
    wgsize_configs = []
    for wgx in [2**i for i in range(11)]:
      for wgy in [2**i for i in range(11)]:
        cfg = (wgx, wgy)
        if self.valid(cfg) and self.sensible(cfg):
          wgsize_configs.append(cfg)
    print('(%s configurations available)' % len(wgsize_configs))

    # Select num_tests evenly spaced wgsize configurations
    if len(wgsize_configs) > num_tests:
      wgsize_configs = self.select_uniform(wgsize_configs, num_tests)

    # Evaluate wgsize configurations
    for cfg in wgsize_configs:
      self.evaluate(cfg, iterations)

  # Check whether a wgsize configuration is valid or not
  def valid(self, wgsize_config):
    if 0 in wgsize_config:
      return False
    if (wgsize_config[0] * wgsize_config[1]) > self.max_wgsize:
      return False
    if self.norder % wgsize_config[0]:
      return False
    if self.norder % wgsize_config[1]:
      return False
    if (self.norder / wgsize_config[0]) % self.config['unroll']:
      return False
    if not self.config['use_wgsize_x'] and wgsize_config[0] > 1:
      return False
    if not self.config['use_wgsize_y'] and wgsize_config[1] > 1:
      return False

    return True


def RunJacobiBenchmark(config,
                       norder,
                       iterations,
                       datatype,
                       device,
                       convergence_frequency=0,
                       convergence_tolerance=0.001,
                       tune_wgsize=False,
                       max_error=0.0,
                       max_runtime=0.0):
  global h_A
  global h_b

  # Print configuration
  SEPARATOR = '--------------------------------'
  print(SEPARATOR)
  print('MATRIX     = %dx%d ' % (norder, norder))
  print('ITERATIONS = %d' % iterations)
  print('DATATYPE   = %s' % datatype)
  if convergence_frequency:
    print('Check convergence every %d iterations (tolerance=%g)' %
          (convergence_frequency, convergence_tolerance))
  else:
    print('Convergence checking disabled')
  print(SEPARATOR)
  print('Work-group size    = ' + str(config['wgsize']))
  print('Unroll factor      = ' + str(config['unroll']))
  print('Data layout        = ' + config['layout'])
  print('Conditional        = ' + config['conditional'])
  print('fmad               = ' + config['fmad'])
  print('Divide by A        = ' + config['divide_A'])
  print('b address space    = ' + config['addrspace_b'])
  print('xold address space = ' + config['addrspace_xold'])
  print('Integer type       = ' + config['integer'])
  print('Relaxed math       = ' + str(config['relaxed_math']))
  print('Use restrict       = ' + str(config['use_restrict']))
  print('Use const pointers = ' + str(config['use_const']))
  print('Use mad24          = ' + str(config['use_mad24']))
  print('Constant norder    = ' + str(config['const_norder']))
  print('Constant wgsize    = ' + str(config['const_wgsize']))
  print('Coalesce columns   = ' + str(config['coalesce_cols']))
  print(SEPARATOR)

  if datatype == 'float':
    dtype = numpy.dtype(numpy.float32)
  elif datatype == 'double':
    dtype = numpy.dtype(numpy.float64)
  else:
    print('Invalid data-type')
    exit(1)

  # Initialize input data
  numpy.random.seed(0)
  h_A = numpy.random.rand(norder, norder).astype(dtype)
  for row in range(norder):
    h_A[row][row] += numpy.sum(h_A[row])
  h_b = numpy.random.rand(norder).astype(dtype)

  # Initialize OpenCL context
  if device:
    context = CL.Context([device])
  else:
    context = CL.create_some_context()
  print('Using \'' + context.devices[0].name + '\'')

  if tune_wgsize:
    tuner = Tuner(config, norder, datatype, context, max_error, max_runtime)

    # Run uniform search with small number of iterations
    tuner.max_error = 0
    short_iterations = max(iterations / 100, 1)
    tuner.uniform_search(100, short_iterations)
    best = tuner.best
    if not best:
      print('No valid configuration found')
      exit(1)
    print('Best work-group size = %s' % (best[0],))
    print('Best runtime = %.4gs (%d iterations)' % (best[1], best[3]))
    print(SEPARATOR)

    # Run local search around best result, at full iteration count
    tuner.reset()
    tuner.max_error = max_error
    tuner.local_search(best[0], iterations)
    print(SEPARATOR)
    best = tuner.best
    if not best:
      print('No valid configuration found')
      exit(1)
    print('Best work-group size = %s' % (best[0],))
    print('Runtime = %.4gs (%d iterations)' % (best[1], best[3]))
    print('Error   = %f' % best[2])
  else:
    try:
      result = RunJacobiBenchmarkConfig(
          config, norder, iterations, datatype, context, max_runtime,
          convergence_frequency, convergence_tolerance)
      print('Runtime = %.4gs (%d iterations)' % (result[0], result[2]))
      print('Error   = %f' % result[1])
      if max_error > 0 and not result[1] < max_error:
        raise 'error too high'
    except Exception as e:
      print('Error: %s' % str(e))
      exit(1)


def RunJacobiBenchmarkConfig(config, norder, iterations, datatype, context,
                             max_runtime, convergence_frequency,
                             convergence_tolerance):
  global h_A
  global h_b

  # Ensure work-group size is valid
  if config['wgsize'][0] & (config['wgsize'][0] - 1):
    raise ValueError('Invalid wgsize[0] value (must be power of two)')
  if norder % config['wgsize'][1]:
    raise ValueError('Invalid wgsize[1] value (must divide matrix order)')

  queue = CL.CommandQueue(context)

  # Create and build program
  build_options = ''
  build_options += ' -cl-fast-relaxed-math' if config['relaxed_math'] else ''
  if config['const_norder']:
    build_options += ' -Dnorder=' + str(norder)
    if config['integer'] == 'uint':
      build_options += 'u'
  kernel_source = GenerateJacobiOpenClKernelSource(config, norder, datatype)
  program = CL.Program(context, kernel_source).build(build_options)

  if datatype == 'float':
    dtype = numpy.dtype(numpy.float32)
  elif datatype == 'double':
    dtype = numpy.dtype(numpy.float64)
  else:
    print('Invalid data-type')
    exit(1)

  # Create buffers
  typesize = dtype.itemsize
  vectorsize = norder * typesize
  matrixsize = norder * norder * typesize
  d_A = CL.Buffer(context, CL.mem_flags.READ_ONLY, size=matrixsize)
  d_b = CL.Buffer(context, CL.mem_flags.READ_ONLY, size=vectorsize)
  d_x0 = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=vectorsize)
  d_x1 = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=vectorsize)
  d_xold = d_x0
  d_xnew = d_x1

  # Initialize data
  h_x = numpy.zeros(norder, dtype)
  CL.enqueue_copy(queue, d_A, h_A)
  CL.enqueue_copy(queue, d_b, h_b)
  CL.enqueue_copy(queue, d_xold, h_x)

  # Create kernels
  jacobi = program.jacobi
  transpose = program.transpose
  precompute_inv_A = program.precompute_inv_A
  convergence = program.convergence

  # Calculate argument indices
  arg_index = 0
  if not config['const_norder']:
    jacobi.set_arg(arg_index, numpy.uint32(norder))
    arg_index += 1
  arg_xold = arg_index
  arg_index += 1
  arg_xnew = arg_index
  arg_index += 1
  arg_A = arg_index
  arg_index += 1
  arg_b = arg_index
  arg_index += 1

  # Compute global size
  local_size = (config['wgsize'][0], config['wgsize'][1])
  global_size = (local_size[0], norder)
  if config['wgsize'][0] > 1:
    jacobi.set_arg(arg_index,
                   CL.LocalMemory(local_size[0] * local_size[1] * typesize))
    arg_index += 1

  # Prepare convergence checking kernel
  conv_wgsize = 64  # TODO: Pick something else? (e.g wgsize[0]*wgsize[1])
  num_groups = norder // conv_wgsize
  h_err = numpy.zeros(num_groups)
  d_err = CL.Buffer(context,
                    CL.mem_flags.WRITE_ONLY,
                    size=num_groups * typesize)
  convergence.set_arg(0, d_x0)
  convergence.set_arg(1, d_x1)
  convergence.set_arg(2, d_err)
  convergence.set_arg(3, CL.LocalMemory(conv_wgsize * typesize))

  with timeout(max_runtime):
    # Start timing
    queue.finish()
    start = time.time()

    if config['layout'] == 'col-major':
      # Run kernel to transpose data on device
      d_A_colmaj = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=matrixsize)
      transpose.set_arg(0, d_A)
      transpose.set_arg(1, d_A_colmaj)
      CL.enqueue_nd_range_kernel(queue, transpose, (norder, norder), None)
      d_A = d_A_colmaj

    if config['divide_A'] in ['precompute-global', 'precompute-constant']:
      # Run kernel to precompute 1/A for diagonal
      d_inv_A = CL.Buffer(context, CL.mem_flags.READ_WRITE, size=vectorsize)
      precompute_inv_A.set_arg(0, d_A)
      precompute_inv_A.set_arg(1, d_inv_A)
      CL.enqueue_nd_range_kernel(queue, precompute_inv_A, (norder,), None)
      jacobi.set_arg(arg_index, d_inv_A)
      arg_index += 1

    jacobi.set_arg(arg_A, d_A)
    jacobi.set_arg(arg_b, d_b)

    # Run Jacobi solver
    for i in range(iterations):
      jacobi.set_arg(arg_xold, d_xold)
      jacobi.set_arg(arg_xnew, d_xnew)
      CL.enqueue_nd_range_kernel(queue, jacobi, global_size, local_size)

      # Convergence check
      if convergence_frequency and (i + 1) % convergence_frequency == 0:
        CL.enqueue_nd_range_kernel(queue, convergence, (norder,),
                                   (conv_wgsize,))
        CL.enqueue_copy(queue, h_err, d_err)
        queue.finish()
        if math.sqrt(numpy.sum(h_err)) < convergence_tolerance:
          break

      d_xold, d_xnew = d_xnew, d_xold

      # TODO(cec): This is the stable point.

    # Stop timing
    queue.finish()
    end = time.time()

  # Read results
  CL.enqueue_copy(queue, h_x, d_xold)

  # Print runtime and final error
  runtime = (end - start)
  error = math.sqrt(sum([e * e for e in (h_b - numpy.dot(h_A, h_x))]))
  return JacobiBenchmarkRun(runtime=runtime, error=error, iteration_count=i + 1)


def GenerateJacobiOpenClKernelSource(config, norder, datatype: str) -> str:
  """Produce the OpenCL kernel source.

  Args:
    config: The configuration dictionary.
    norder: The matrix size as an integer.
    datatype: The name of the dataype, either double or float.

  Returns:
    A string.
  """

  def gen_ptrarg(config, addrspace, name, readonly=True):
    const = 'const' if readonly and config['use_const'] else ''
    restrict = 'restrict' if config['use_restrict'] else ''
    ptrarg = '%-8s %-5s %s *%s %s'
    return ptrarg % (addrspace, const, datatype, restrict, name)

  def gen_index(config, col, row, N):
    if config['layout'] == 'row-major':
      x, y = col, row
    elif config['layout'] == 'col-major':
      x, y = row, col
    else:
      raise ValueError('layout', 'must be \'row-major\' or \'col-major\'')

    if config['use_mad24']:
      return 'mad24(%s, %s, %s)' % (y, N, x)
    else:
      return '(%s*%s + %s)' % (y, N, x)

  def gen_fmad(config, x, y, z):
    if config['fmad'] == 'op':
      return '(%s * %s + %s)' % (x, y, z)
    elif config['fmad'] == 'fma':
      return 'fma(%s, %s, %s)' % (x, y, z)
    elif config['fmad'] == 'mad':
      return 'mad(%s, %s, %s)' % (x, y, z)
    else:
      raise ValueError('fmad', 'must be \'op\' or \'fma\' or \'mad\')')

  def gen_cond_accum(config, cond, acc, a, b):
    result = ''
    if config['conditional'] == 'branch':
      result += 'if (%s) ' % cond
      _b = b
    elif config['conditional'] == 'mask':
      _b = '%s*(%s)' % (b, cond)
    else:
      raise ValueError('conditional', 'must be \'branch\' or \'mask\'')
    result += '%s = %s' % (acc, gen_fmad(config, a, _b, acc))
    return result

  def gen_divide_A(config, numerator):
    index = gen_index(config, 'row', 'row', 'norder')
    if config['divide_A'] == 'normal':
      return '(%s) / A[%s]' % (numerator, index)
    elif config['divide_A'] == 'native':
      return 'native_divide(%s, A[%s])' % (numerator, index)
    elif config['divide_A'] in ['precompute-global', 'precompute-constant']:
      return '(%s) * inv_A[row]' % numerator
    else:
      raise ValueError(
          'divide_A', 'must be \'normal\' or \'native\' or ' +
          '\'precompute-global\' or \'precompute-constant\'')

  # Ensure addrspace_b value is valid
  if not config['addrspace_b'] in ['global', 'constant']:
    raise ValueError('addrspace_b', 'must be \'global\' or \'constant\'')

  # Ensure addrspace_xold value is valid
  if not config['addrspace_xold'] in ['global', 'constant']:
    raise ValueError('addrspace_xold', 'must be \'global\' or \'constant\'')

  # Ensure integer value is valid
  if not config['integer'] in ['uint', 'int']:
    raise ValueError('integer', 'must be \'uint\' or \'or\'')
  inttype = str(config['integer'])

  # Ensure unroll factor is valid
  cols_per_wi = norder / config['wgsize'][0]
  if cols_per_wi % config['unroll']:
    print('Invalid unroll factor (must exactly divide %d)' % cols_per_wi)
    exit(1)

  row = 'get_global_id(1)'

  lidx = 'get_local_id(0)'
  lidy = 'get_local_id(1)'
  lszx = 'get_local_size(0)'
  lszy = 'get_local_size(1)'
  if config['const_wgsize']:
    if config['wgsize'][0] == 1:
      lidx = '0'
    if config['wgsize'][1] == 1:
      lidy = '0'
    lszx = config['wgsize'][0]
    lszy = config['wgsize'][1]

  result = ''

  # Enable FP64 extension for OpenCL 1.1 devices
  if datatype == 'double':
    result += '\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n'

  result += '\n kernel void jacobi('

  # Kernel arguments
  if not config['const_norder']:
    result += '\n  const %s norder,' % inttype
  result += '\n  %s,' % gen_ptrarg(config, config['addrspace_xold'], 'xold')
  result += '\n  %s,' % gen_ptrarg(config, 'global', 'xnew', False)
  result += '\n  %s,' % gen_ptrarg(config, 'global', 'A')
  result += '\n  %s,' % gen_ptrarg(config, config['addrspace_b'], 'b')
  if config['wgsize'][0] > 1:
    result += '\n  %s,' % gen_ptrarg(config, 'local', 'scratch', False)
  if config['divide_A'] == 'precompute-global':
    result += '\n  %s,' % gen_ptrarg(config, 'global', 'inv_A')
  elif config['divide_A'] == 'precompute-constant':
    result += '\n  %s,' % gen_ptrarg(config, 'constant', 'inv_A')
  result = result[:-1]
  result += ')'

  # Start of kernel
  result += '\n{'

  # Get row index
  result += '\n  const %s row  = %s;' % (inttype, row)
  result += '\n  const %s lidx = %s;' % (inttype, lidx)
  result += '\n  const %s lszx = %s;' % (inttype, lszx)

  # Get column range for work-item
  if config['coalesce_cols']:
    col_beg = 'lidx'
    col_end = 'norder'
    col_inc = 'lszx'
  else:
    col_beg = 'lidx*%d' % cols_per_wi
    col_end = '%s+%d' % (col_beg, cols_per_wi)
    col_inc = '1'

  # Initialise accumulator
  result += '\n\n  %s tmp = 0.0;' % datatype

  # Loop begin
  result += '\n  for (%s col = %s; col < %s; )' % (inttype, col_beg, col_end)
  result += '\n  {'

  # Loop body
  A = 'A[%s]' % gen_index(config, 'col', 'row', 'norder')
  x = 'xold[col]'
  loop_body = '\n    %s;' % gen_cond_accum(config, 'row != col', 'tmp', A, x)
  loop_body += '\n    col += %s;' % col_inc
  result += loop_body * config['unroll']

  # Loop end
  result += '\n  }\n'

  # xnew = (b - tmp) / D
  if config['wgsize'][0] > 1:
    result += '\n  int lid = %s + %s*%s;' % (lidx, lidy, lszx)
    result += '\n  scratch[lid] = tmp;'
    result += '\n  barrier(CLK_LOCAL_MEM_FENCE);'
    result += '\n  for (%s offset = lszx/2; offset>0; offset/=2)' % inttype
    result += '\n  {'
    result += '\n    if (lidx < offset)'
    result += '\n      scratch[lid] += scratch[lid + offset];'
    result += '\n    barrier(CLK_LOCAL_MEM_FENCE);'
    result += '\n  }'
    result += '\n  if (lidx == 0)'
    xnew = gen_divide_A(config, 'b[row] - scratch[lid]')
    result += '\n    xnew[row] = %s;' % xnew
  else:
    xnew = gen_divide_A(config, 'b[row] - tmp')
    result += '\n  xnew[row] = %s;' % xnew

  # End of kernel
  result += '\n}\n'

  # Convergence checking kernel
  result += '''
kernel void transpose(global %(datatype)s *input, global %(datatype)s *output)
{
  int row = get_global_id(0);
  int col = get_global_id(1);
  int n   = get_global_size(0);
  output[row*n + col] = input[col*n + row];
}

kernel void precompute_inv_A(global %(datatype)s *A, global %(datatype)s *inv_A)
{
  int row = get_global_id(0);
  int n   = get_global_size(0);
  inv_A[row] = 1 / A[row*n + row];
}

kernel void convergence(global %(datatype)s *x0,
                        global %(datatype)s *x1,
                        global %(datatype)s *result,
                        local  %(datatype)s *scratch)
{
  uint row = get_global_id(0);
  uint lid = get_local_id(0);
  uint lsz = get_local_size(0);

  %(datatype)s diff = x0[row] - x1[row];
  scratch[lid] = diff*diff;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint offset = lsz/2; offset > 0; offset/=2)
  {
    if (lid < offset)
      scratch[lid] += scratch[lid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    result[get_group_id(0)] = scratch[0];
}
    ''' % {
      'datatype': datatype
  }

  return str(result)


def get_device_list():
  platforms = CL.get_platforms()
  devices = []
  for p in platforms:
    devices += p.get_devices()
  return devices


def main():
  # Command-line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--norder', type=int, default=256)
  parser.add_argument('-i', '--iterations', type=int, default=1000)
  parser.add_argument('-f',
                      '--datatype',
                      choices=['float', 'double'],
                      default='double')
  parser.add_argument('-c', '--config', default='')
  parser.add_argument('-k', '--convergence-frequency', type=int, default=0)
  parser.add_argument('-t',
                      '--convergence-tolerance',
                      type=float,
                      default=0.001)
  parser.add_argument('-p', '--print-kernel', action='store_true')
  parser.add_argument('-l', '--list', action='store_true')
  parser.add_argument('-d', '--device', type=int, default=0)
  parser.add_argument('--tune-wgsize', action='store_true')
  parser.add_argument('--max-error', type=float, default=0)
  parser.add_argument('--max-runtime', type=int, default=3600)
  args = parser.parse_args()

  # Print device list
  if args.list:
    devices = get_device_list()
    if devices:
      print
      print('OpenCL devices:')
      for i in range(len(devices)):
        print('  %d: %s' % (i, devices[i].name))
      print
    else:
      print('No OpenCL devices found')
    exit(0)

  # Default configuration
  config = dict()
  config['wgsize'] = [64, 1]
  config['unroll'] = 1
  config['layout'] = 'row-major'
  config['conditional'] = 'branch'
  config['fmad'] = 'op'
  config['divide_A'] = 'normal'
  config['addrspace_b'] = 'global'
  config['addrspace_xold'] = 'global'
  config['integer'] = 'uint'
  config['relaxed_math'] = False
  config['use_const'] = False
  config['use_restrict'] = False
  config['use_mad24'] = False
  config['const_norder'] = False
  config['const_wgsize'] = False
  config['coalesce_cols'] = True

  # Load config from JSON file
  if args.config:
    with open(args.config) as config_file:
      config.update(json.load(config_file))

  if args.print_kernel:
    print(GenerateJacobiOpenClKernelSource(config, args.norder, args.datatype))
    exit(0)

  # Run Jacobi solver
  device = get_device_list()[args.device]
  RunJacobiBenchmark(config, args.norder, args.iterations, args.datatype,
                     device, args.convergence_frequency,
                     args.convergence_tolerance, args.tune_wgsize,
                     args.max_error, args.max_runtime)


if __name__ == '__main__':
  app.Run(main)
