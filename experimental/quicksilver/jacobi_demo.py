"""Demonstration of heterogeneous device pre-empting to maximise performance.

This script simulates two heterogeneous workloads, app A and app B, where:
 * App A has a modest performance improvement on GPU.
 * App B has a significant performance improvement on GPU.
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

    Speedup of App A on GPU: 1.827975009040212
    Speedup of App B on GPU: 8.04094950510637

    Approach A: Immediately launch app on best available device
    Approach A in 1m 2s 241ms
    Approach A: App A: runtime=34.825s, iterations=1,000,001, throughput=28.7k iterations / second
    Approach A: App B: runtime=60.199s, iterations=100,001, throughput=1.7k iterations / second

    Approach B: Wait for best-available device
    section 1 in 15s 263ms
    section 2 in 7s 306ms
    Approach B in 22s 570ms
    Approach B: App A: runtime=15.262s, iterations=1,000,001, throughput=65.5k iterations / second
    Approach B: App B: runtime=7.289s, iterations=100,001, throughput=13.7k iterations / second

    Approach C: Pre-empt lower speedup job
    section 1 in 2s 27ms
    section 2 in 7s 285ms
    section 3 in 10s 640ms
    Approach C in 19s 956ms
    Approach C: App A1: runtime=2.007s, iterations=132,142, throughput=65.8k iterations / second
    Approach C: App A2: runtime=7.294s, iterations=174,987, throughput=24.0k iterations / second
    Approach C: App A3: runtime=10.640s, iterations=692,872, throughput=65.1k iterations / second
    Approach C total iterations: 1000001
    Approach C: App B: runtime=7.296s, iterations=100,001, throughput=13.7k iterations / second
"""

import json
import time

import pyopencl as CL

from datasets.benchmarks.jacobi_opencl import jacobi_opencl as jacobi
from labm8 import app
from labm8 import jsonutil
from labm8 import humanize
from labm8 import bazelutil
from labm8 import prof

FLAGS = app.FLAGS

APP_A_CONFIG = bazelutil.DataPath(
    'phd/experimental/quicksilver/app_a_jacobi_config.json')
APP_B_CONFIG = bazelutil.DataPath(
    'phd/experimental/quicksilver/app_b_jacobi_config.json')


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


def ApproachB(app_a_config, app_b_config, gpu, cpu, app_b_delay):
  print('Approach B: Wait for best-available device')
  a = jacobi.JacobiBenchmarkThread(app_a_config, gpu)
  b = jacobi.JacobiBenchmarkThread(app_b_config, gpu)
  with prof.ProfileToStdout('Approach B'):
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


def ApproachC(app_a_config, app_b_config, gpu, cpu, app_b_delay):
  print('Approach C: Pre-empt lower speedup job')
  a1 = jacobi.JacobiBenchmarkThread(app_a_config, gpu)
  a2 = jacobi.JacobiBenchmarkThread(app_a_config, cpu)
  a3 = jacobi.JacobiBenchmarkThread(app_a_config, gpu)
  b = jacobi.JacobiBenchmarkThread(app_b_config, gpu)
  with prof.ProfileToStdout('Approach C'):
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


def main():
  """Main entry point."""
  app_a_config = jsonutil.read_file(APP_A_CONFIG)
  app_b_config = jsonutil.read_file(APP_B_CONFIG)

  gpu, cpu = GetGpuDevice(), GetCpuDevice()
  app_b_delay = 2

  app_a_speedup = GetGpuSpeedup(app_a_config, gpu, cpu)
  app_b_speedup = GetGpuSpeedup(app_b_config, gpu, cpu)

  print(f'Speedup of App A on GPU: {app_a_speedup}')
  print(f'Speedup of App B on GPU: {app_b_speedup}')

  print()
  ApproachA(app_a_config, app_b_config, gpu, cpu, app_b_delay)
  ApproachB(app_a_config, app_b_config, gpu, cpu, app_b_delay)
  ApproachC(app_a_config, app_b_config, gpu, cpu, app_b_delay)


if __name__ == '__main__':
  app.Run(main)
