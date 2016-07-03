import re
import os

from tempfile import mkstemp
from shutil import move
from os import remove, close

# parboil compile [benchmark] opencl_base
# parboil run [benchmark] [dataset]

# Parboil parallel benchmark suite, version 0.2

# Resolving OpenCL library...
#         libOpenCL.so.1 => /usr/lib64/libOpenCL.so.1 (0x00007f2e2adfd000)
# Number of Platforms found: 1
#   Number of Devices found for Platform 0: 1
# Chose Device Type: GPU
# Starting GPU kernel
# GPU kernel done
# IO        : 0.870675
# Kernel    : 0.174193
# Copy      : 0.724049
# Driver    : 0.174197
# CPU/Kernel Overlap: 0.174211
# Timer Wall Time: 1.768981
# Pass

# ./parboil run spmv opencl_base small
# ./parboil run spmv opencl_base medium
# ./parboil run spmv opencl_base large

class BenchmarkException(Exception): pass

class Device(object):
    GPU = "CL_DEVICE_TYPE_GPU"
    CPU = "CL_DEVICE_TYPE_CPU"


class kernel_file(object):
    def __init__(self, path):
        path = os.path.expanduser(path)

        # Check that file exists
        if not os.path.exists(path):
            raise KernelException("Parboil kernel file '{}' not found"
                                  .format(path))

        with open(path) as infile:
            contents = infile.read()

        # Set member variables
        self.path = path
        self.contents = contents


def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(re.sub(pattern, subst, line))
    close(fh)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)


class implementation_file(object):
    def __init__(self, path):
        path = os.path.expanduser(path)

        # Check that file exists
        if not os.path.exists(path):
            raise KernelException("Parboil implementation file '{}' not found"
                                  .format(path))

        # Set member variables
        self.path = path

    def set_device_type(self, devtype):
        replace(self.path,
                re.compile('#define MY_DEVICE_TYPE .+'),
                '#define MY_DEVICE_TYPE ' + str(devtype))


class benchmark(object):
    def __init__(self, parboil_root, name):
        # Check that parboil exists
        parboil_root = os.path.expanduser(parboil_root)
        if not os.path.exists(parboil_root):
            raise BenchmarkException("Parboil root '{}' not found"
                                     .format(parboil_root))

        # Check that benchmark exists
        if not os.path.exists(os.path.join(parboil_root, 'benchmarks', name)):
            raise BenchmarkException("Parboil benchmark '{}' not found"
                                     .format(name))

        # Check that opencl_base or equivalent exists
        implementation = 'opencl_base'
        if not os.path.exists(os.path.join(
                parboil_root, 'benchmarks', name, 'src', implementation)):
            raise BenchmarkException("Parboil implementation '{}' not found"
                                     .format(implementation))

        src_file = implementation_file(
            os.path.join(parboil_root, 'benchmarks', name,
                         'src', implementation, 'main.c'))

        # Get datasets
        datasets_path = os.path.join(parboil_root, 'datasets', name)
        datasets = os.listdir(datasets_path)

        # Get OpenCL kernel
        kernel_file_path = os.path.join(parboil_root, 'benchmarks', name,
                                        'src', implementation, 'kernel.cl')
        if not os.path.exists(kernel_file_path):
            raise BenchmarkException("Parboil OpenCL kernel '{}' not found"
                                     .format(kernel_file_path))

        # Set member variables
        self.parboil_root = parboil_root
        self.name = name
        self.implementation = implementation
        self.datasets = datasets
        self.kernel_file_path = kernel_file_path
        self.src_file = src_file


    def run(self, dataset, device_type=Device.GPU, n=1):
        if dataset not in self.datasets:
            raise BenchmarkException("No such dataset '{}'".format(dataset))

        self.src_file.set_device_type(device_type)

        cmd = ("./parboil compile {} {}"
               .format(benchmark, self.implementation))

        cmd = ("./parboil run {} {} {}"
               .format(benchmark, self.implementation, dataset))
        # TODO: Process output, insert into db


    def __repr__(self):
        return "{}: {}".format(self.name, ", ".join(self.datasets))
