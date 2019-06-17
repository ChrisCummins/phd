"""Demonstration of heterogeneous device pre-empting to maximise performance.

This script simulates two heterogeneous workloads, app A and app B, where:
 * App A is a small, long-running job that earns a modest performance
   improvement on GPU over the CPU.
 * App B is a large, short-running job that has a significant performance
   improvement on GPU over the CPU.
 * App B launches 2 seconds after App A.

Both apps are iterative Jacobi stencil patterns.

This script profiles three approaches to scheduling these two apps. In all
three approaches, App A begins on the GPU since it provides the best
performance. The three approaches to scheduling app B are:

 A. Immediately start App B using the CPU, since the CPU is free.
 B. Wait for App A to finish with the GPU, then start App B on the GPU.
 C. Use "stable points" in pattern execution to wait for an opportunity to
    safely interrupt App A, copying the results back to the CPU and resuming
    work on the CPU. App B may now fully utilise the GPU. Once App B completes,
    App A is free to move back to the GPU at the next stable point.

The idea is that, by selecting the program which will best utilise the GPU,
approach C provides the best utilisation of available resources, despite
incurring increased costs of parallel execution and additional memory
transfers.

Reference run:

    $ bazel run //experimental/quicksilver:jacobi_demo
    Using GPU: GeForce GTX 1080
    Using CPU: Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz

    Speedup of App A on GPU: 1.4958583971925479
    Speedup of App B on GPU: 9.966833963065307

    Approach A: Immediately launch app on best available device
    Approach A in 4m 57s 962ms
    Approach A: App A: runtime=42.857s, iterations=1,000,001, throughput=23.3k iterations / second
    Approach A: App B: runtime=295.834s, iterations=400,001, throughput=1.4k iterations / second

    Approach B: Wait for best-available device
    section 1 in 16s 156ms
    section 2 in 29s 327ms
    Approach B in 45s 483ms
    Approach B: App A: runtime=16.156s, iterations=1,000,001, throughput=61.9k iterations / second
    Approach B: App B: runtime=29.326s, iterations=400,001, throughput=13.6k iterations / second

    Approach C: Pre-empt lower speedup job
    section 1 in 2s 33ms
    section 2 in 29s 514ms
    section 3 in 3s 637ms
    Approach C in 35s 199ms
    Approach C: App A1: runtime=2.007s, iterations=126,555, throughput=63.1k iterations / second
    Approach C: App A2: runtime=29.539s, iterations=648,200, throughput=21.9k iterations / second
    Approach C: App A3: runtime=3.637s, iterations=225,246, throughput=61.9k iterations / second
    Approach C total iterations: 1000001
    Approach C: App B: runtime=29.523s, iterations=400,001, throughput=13.5k iterations / second

    Speedup of approach C over approach B: 1.292x
    bazel run experimental/quicksilver:jacobi_demo  6136.79s user 166.72s system 1115% cpu 9:25.27 total
"""

import json
import time

import pyopencl as CL

from datasets.benchmarks.jacobi_opencl import jacobi_opencl as jacobi
from labm8 import app
from labm8 import jsonutil
from labm8 import humanize
from labm8 import prof

FLAGS = app.FLAGS

# App A is a small, long running job which has a modest speedup on GPU.
APP_A_CONFIG = {
    "norder": 64,
    "iteration_count": 1000000,
    "datatype": "double",
    "convergence_frequency": 0,
    "convergence_tolerance": 0.001,
    "wgsize": [64, 1],
    "unroll": 1,
    "layout": "row-major",
    "conditional": "branch",
    "fmad": "op",
    "divide_A": "normal",
    "addrspace_b": "global",
    "addrspace_xold": "global",
    "integer": "uint",
    "relaxed_math": False,
    "use_const": False,
    "use_restrict": False,
    "use_mad24": False,
    "const_norder": False,
    "const_wgsize": False,
    "coalesce_cols": True,
    "min_runtime": 0,
    "max_runtime": 0
}

# App B is a large, short running job which has a high speedup on GPU.
APP_B_CONFIG = {
    "norder": 1024,
    "iteration_count": 400000,
    "datatype": "float",
    "convergence_frequency": 0,
    "convergence_tolerance": 0.001,
    "wgsize": [32, 1],
    "unroll": 1,
    "layout": "col-major",
    "conditional": "branch",
    "fmad": "op",
    "divide_A": "normal",
    "addrspace_b": "global",
    "addrspace_xold": "global",
    "integer": "int",
    "relaxed_math": False,
    "use_const": False,
    "use_restrict": False,
    "use_mad24": False,
    "const_norder": False,
    "const_wgsize": False,
    "coalesce_cols": True,
    "min_runtime": 0,
    "max_runtime": 0
}


def GetGpuDevice() -> CL.Device:
  for device in jacobi.GetDeviceList():
    if device.type == CL.device_type.GPU:
      app.Log(1, 'Using GPU: %s', device.name)
      return device


def GetCpuDevice() -> CL.Device:
  for device in jacobi.GetDeviceList():
    if device.type == CL.device_type.CPU:
      app.Log(1, 'Using CPU: %s', device.name)
      return device


def GetGpuSpeedup(config, gpu, cpu) -> float:
  """Measure the speedup of configuration on GPU over CPU."""
  benchmark_config = config.copy()
  benchmark_config['iteration_count'] = 10
  benchmark_config['min_runtime'] = 5
  benchmark_config['max_runtime'] = 5

  gpu_run = jacobi.RunJacobiBenchmark(benchmark_config, gpu)
  cpu_run = jacobi.RunJacobiBenchmark(benchmark_config, cpu)

  return gpu_run.throughput / cpu_run.throughput


def Stringify(a):
  return f"runtime={a.runtime:.3f}s, iterations={humanize.Commas(a.iteration_count)}, throughput={a.throughput / 1000:.1f}k iterations / second"


def ApproachA(app_a_config, app_b_config, gpu, cpu, app_b_delay):
  print('Approach A: Immediately launch app on best available device')
  a = jacobi.JacobiBenchmarkThread(app_a_config, gpu)
  b = jacobi.JacobiBenchmarkThread(app_b_config, cpu)
  with prof.ProfileToStdout('Approach A'):
    start = time.time()
    # Run App A on GPU.
    a.start()
    time.sleep(app_b_delay)
    # Run App B on CPU.
    b.start()
    a.join()
    b.join()
  print(f'Approach A: App A: {Stringify(a.GetResult())}')
  print(f'Approach A: App B: {Stringify(b.GetResult())}')
  print()
  return time.time() - start


def ApproachB(app_a_config, app_b_config, gpu, cpu, app_b_delay):
  print('Approach B: Wait for best-available device')
  a = jacobi.JacobiBenchmarkThread(app_a_config, gpu)
  b = jacobi.JacobiBenchmarkThread(app_b_config, gpu)
  with prof.ProfileToStdout('Approach B'):
    start = time.time()

    # Run App A on GPU.
    with prof.ProfileToStdout('section 1'):
      a.start()
      time.sleep(app_b_delay)
      a.join()

    # Then run App B on GPU.
    with prof.ProfileToStdout('section 2'):
      b.start()
      b.join()
  print(f'Approach B: App A: {Stringify(a.GetResult())}')
  print(f'Approach B: App B: {Stringify(b.GetResult())}')
  print()
  return time.time() - start


def ApproachC(app_a_config, app_b_config, gpu, cpu, app_b_delay):
  print('Approach C: Pre-empt lower speedup job')
  a1 = jacobi.JacobiBenchmarkThread(app_a_config, gpu)
  a2 = jacobi.JacobiBenchmarkThread(app_a_config, cpu)
  a3 = jacobi.JacobiBenchmarkThread(app_a_config, gpu)
  b = jacobi.JacobiBenchmarkThread(app_b_config, gpu)
  with prof.ProfileToStdout('Approach C'):
    start = time.time()

    # Start App A on GPU.
    a1.start()

    # App B is going to pre-empt App A on GPU. App A moves over to the CPU.
    with prof.ProfileToStdout('section 1'):
      time.sleep(app_b_delay)
      a1.Interrupt()
      a1.join()
      a1_result = a1.GetResult()

      a2._config['iteration_count'] -= a1_result.iteration_count
      b.start()
      a2.start()

    # Block until App B is done.
    with prof.ProfileToStdout('section 2'):
      b.join()
      a2.Interrupt()
      a2.join()
      a2_result = a2.GetResult()

    # App A now moves back to the GPU until done.
    with prof.ProfileToStdout('section 3'):
      a3._config['iteration_count'] -= a2_result.iteration_count
      a3.start()
      a3.join()

  print(f'Approach C: App A1: {Stringify(a1_result)}')
  print(f'Approach C: App A2: {Stringify(a2_result)}')
  print(f'Approach C: App A3: {Stringify(a3.GetResult())}')
  print(
      'Approach C total iterations:',
      str(a1_result.iteration_count + a2_result.iteration_count +
          a3.GetResult().iteration_count))
  print(f'Approach C: App B: {Stringify(b.GetResult())}')
  print()
  return time.time() - start


def main():
  """Main entry point."""
  gpu, cpu = GetGpuDevice(), GetCpuDevice()
  # The number of seconds after the start of App A before launching App B.
  app_b_delay = 2

  app_a_speedup = GetGpuSpeedup(APP_A_CONFIG, gpu, cpu)
  app_b_speedup = GetGpuSpeedup(APP_B_CONFIG, gpu, cpu)

  print(f'Speedup of App A on GPU: {app_a_speedup}')
  print(f'Speedup of App B on GPU: {app_b_speedup}')

  print()
  a = ApproachA(APP_A_CONFIG, APP_B_CONFIG, gpu, cpu, app_b_delay)
  b = ApproachB(APP_A_CONFIG, APP_B_CONFIG, gpu, cpu, app_b_delay)
  c = ApproachC(APP_A_CONFIG, APP_B_CONFIG, gpu, cpu, app_b_delay)

  print(f"Speedup of approach C over approach B: {b / c:.3f}x")


if __name__ == '__main__':
  app.Run(main)
