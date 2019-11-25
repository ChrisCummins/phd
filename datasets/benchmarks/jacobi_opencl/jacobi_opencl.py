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
import argparse
import collections
import json
import math
import signal
import threading
import time
import typing

import numpy as np
import pyopencl as CL

from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_input_path('config', None, 'Path to config file')
app.DEFINE_integer('device_id', 0, 'The device ID to use.')
app.DEFINE_boolean('list_devices', False, 'Print devices and exit.')


class JacobiBenchmarkRun(typing.NamedTuple):
  runtime: float
  error: int
  iteration_count: int
  throughput: float


def GetBuildOptions(config) -> str:
  build_options = ''
  build_options += ' -cl-fast-relaxed-math' if config['relaxed_math'] else ''
  if config['const_norder']:
    build_options += ' -Dnorder=' + str(config['norder'])
    if config['integer'] == 'uint':
      build_options += 'u'
  return build_options


def RunJacobiBenchmark(config, device):
  thread = JacobiBenchmarkThread(config, device)
  thread.start()
  thread.join()
  return thread.GetResult()


class JacobiBenchmarkThread(threading.Thread):

  def __init__(self, config, device):
    super(JacobiBenchmarkThread, self).__init__()

    self._device = device
    self._config = config
    self._runtime = 0
    self._iteration_count = 0
    self._stop_event = threading.Event()
    if self._config['datatype'] == 'float':
      self._dtype = np.dtype(np.float32)
    elif self._config['datatype'] == 'double':
      self._dtype = np.dtype(np.float64)
    else:
      raise LookupError

    # Initialize input data
    np.random.seed(0)
    self._h_A = np.random.rand(self._config['norder'],
                               self._config['norder']).astype(self._dtype)
    for row in range(self._config['norder']):
      self._h_A[row][row] += np.sum(self._h_A[row])
    self._h_b = np.random.rand(self._config['norder']).astype(self._dtype)

    # Initialize OpenCL context
    self._context = CL.Context([self._device])

    # Ensure work-group size is valid
    if config['wgsize'][0] & (config['wgsize'][0] - 1):
      raise ValueError('Invalid wgsize[0] value (must be power of two)')
    if config['norder'] % config['wgsize'][1]:
      raise ValueError('Invalid wgsize[1] value (must divide matrix order)')

    self._queue = CL.CommandQueue(self._context)

    # Create and build program
    build_options = GetBuildOptions(config)
    kernel_source = GenerateJacobiOpenClKernelSource(config)
    program = CL.Program(self._context, kernel_source).build(build_options)

    # Create buffers
    typesize = self._dtype.itemsize
    self._vectorsize = config['norder'] * typesize
    self._matrixsize = config['norder'] * config['norder'] * typesize
    self._d_A = CL.Buffer(self._context,
                          CL.mem_flags.READ_ONLY,
                          size=self._matrixsize)
    self._d_b = CL.Buffer(self._context,
                          CL.mem_flags.READ_ONLY,
                          size=self._vectorsize)
    self._d_x0 = CL.Buffer(self._context,
                           CL.mem_flags.READ_WRITE,
                           size=self._vectorsize)
    self._d_x1 = CL.Buffer(self._context,
                           CL.mem_flags.READ_WRITE,
                           size=self._vectorsize)
    self._d_xold = self._d_x0
    self._d_xnew = self._d_x1

    # Initialize data
    self._h_x = np.zeros(config['norder'], self._dtype)
    CL.enqueue_copy(self._queue, self._d_A, self._h_A)
    CL.enqueue_copy(self._queue, self._d_b, self._h_b)
    CL.enqueue_copy(self._queue, self._d_xold, self._h_x)

    # Create kernels
    self._jacobi = program.jacobi
    self._transpose = program.transpose
    self._precompute_inv_A = program.precompute_inv_A
    self._convergence = program.convergence

    # Calculate argument indices
    arg_index = 0
    if not config['const_norder']:
      self._jacobi.set_arg(arg_index, np.uint32(config['norder']))
      arg_index += 1
    self._arg_xold = arg_index
    arg_index += 1
    self._arg_xnew = arg_index
    arg_index += 1
    self._arg_A = arg_index
    arg_index += 1
    self._arg_b = arg_index
    arg_index += 1

    # Compute global size
    self._local_size = (config['wgsize'][0], config['wgsize'][1])
    self._global_size = (self._local_size[0], config['norder'])
    if config['wgsize'][0] > 1:
      self._jacobi.set_arg(
          arg_index,
          CL.LocalMemory(self._local_size[0] * self._local_size[1] * typesize))
      arg_index += 1

    # Prepare convergence checking kernel
    conv_wgsize = 64  # TODO: Pick something else? (e.g wgsize[0]*wgsize[1])
    num_groups = config['norder'] // conv_wgsize
    h_err = np.zeros(num_groups)
    self._d_err = CL.Buffer(self._context,
                            CL.mem_flags.WRITE_ONLY,
                            size=num_groups * typesize)
    self._convergence.set_arg(0, self._d_x0)
    self._convergence.set_arg(1, self._d_x1)
    self._convergence.set_arg(2, self._d_err)
    self._convergence.set_arg(3, CL.LocalMemory(conv_wgsize * typesize))

  def run(self):
    config = self._config

    # Start timing
    self._queue.finish()
    start = time.time()

    if config['layout'] == 'col-major':
      # Run kernel to self._transpose data on device
      self._d_A_colmaj = CL.Buffer(self._context,
                                   CL.mem_flags.READ_WRITE,
                                   size=self._matrixsize)
      self._transpose.set_arg(0, self._d_A)
      self._transpose.set_arg(1, self._d_A_colmaj)
      CL.enqueue_nd_range_kernel(self._queue, self._transpose,
                                 (config['norder'], config['norder']), None)
      self._d_A = self._d_A_colmaj

    if config['divide_A'] in ['precompute-global', 'precompute-constant']:
      # Run kernel to precompute 1/A for diagonal
      self._d_inv_A = CL.Buffer(self._context,
                                CL.mem_flags.READ_WRITE,
                                size=self._vectorsize)
      self._precompute_inv_A.set_arg(0, self._d_A)
      self._precompute_inv_A.set_arg(1, self._d_inv_A)
      CL.enqueue_nd_range_kernel(self._queue, self._precompute_inv_A,
                                 (config['norder'],), None)
      self._jacobi.set_arg(arg_index, self._d_inv_A)
      arg_index += 1

    self._jacobi.set_arg(self._arg_A, self._d_A)
    self._jacobi.set_arg(self._arg_b, self._d_b)

    # Run Jacobi solver
    i = self._iteration_count
    while not self._stop_event.is_set():
      runtime = self._runtime + time.time() - start
      if ((i > config['iteration_count'] and runtime > config['min_runtime']) or
          (config['max_runtime'] and runtime > config['max_runtime'])):
        break
      self._jacobi.set_arg(self._arg_xold, self._d_xold)
      self._jacobi.set_arg(self._arg_xnew, self._d_xnew)
      CL.enqueue_nd_range_kernel(self._queue, self._jacobi, self._global_size,
                                 self._local_size)

      # Convergence check
      if config['convergence_frequency'] and (
          i + 1) % config['convergence_frequency'] == 0:
        CL.enqueue_nd_range_kernel(self._queue, self._convergence,
                                   (config['norder'],), (conv_wgsize,))
        CL.enqueue_copy(self._queue, h_err, self._d_err)
        queue.finish()
        if math.sqrt(np.sum(h_err)) < config['convergence_tolerance']:
          break

      self._d_xold, self._d_xnew = self._d_xnew, self._d_xold
      i += 1

    # Stop timing
    self._queue.finish()
    end = time.time()

    # Read results
    CL.enqueue_copy(self._queue, self._h_x, self._d_xold)

    # Print runtime and final error
    runtime = (end - start)
    error = math.sqrt(
        sum([e * e for e in (self._h_b - np.dot(self._h_A, self._h_x))]))

    self._runtime += runtime
    self._iteration_count = i
    self._stop_event.clear()

  def Interrupt(self) -> None:
    self._stop_event.set()

  def GetResult(self) -> JacobiBenchmarkRun:
    return JacobiBenchmarkRun(runtime=self._runtime,
                              error=-1,
                              iteration_count=self._iteration_count,
                              throughput=self._iteration_count / self._runtime)


def GenerateJacobiOpenClKernelSource(config) -> str:
  """Produce the OpenCL kernel source.

  Args:
    config: The configuration dictionary.

  Returns:
    A string.
  """
  datatype = config['datatype']

  def gen_ptrarg(config, addrspace, name, readonly=True):
    const = 'const' if readonly and config['use_const'] else ''
    restrict = 'restrict' if config['use_restrict'] else ''
    ptrarg = '%-8s %-5s %s *%s %s'
    return ptrarg % (addrspace, const, config['datatype'], restrict, name)

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
  cols_per_wi = config['norder'] / config['wgsize'][0]
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
  if config['datatype'] == 'double':
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


def GetDeviceList() -> typing.List[CL.Device]:
  platforms = CL.get_platforms()
  devices = []
  for p in platforms:
    devices += p.get_devices()
  return devices


def main():
  # Print device list
  if FLAGS.list_devices:
    devices = GetDeviceList()
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
  with open(FLAGS.config) as config_file:
    config.update(json.load(config_file))

  # Run Jacobi solver
  device = GetDeviceList()[FLAGS.device_id]
  RunJacobiBenchmark(config, device)


if __name__ == '__main__':
  app.Run(main)
