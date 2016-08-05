import smith

import os

from socket import gethostname

import labm8
from labm8 import fs
from labm8 import system


class ConfigException(smith.SmithException): pass


def host_has_opencl():
    try:
        return system.is_mac() or len(fs.ls('/etc/OpenCL/vendors'))
    except FileNotFoundError:
        return False


def host_has_gpu():
    # TODO:
    return host_has_opencl()


def host_has_cpu():
    # TODO:
    return host_has_opencl()


def phd_root():
    return smith.assert_exists("~/phd", exception=ConfigException)


def torch_rnn_path():
    return smith.assert_exists("~/src/torch-rnn", exception=ConfigException)


def parboil_root():
    def verify_parboil(path):
        smith.assert_exists(path, exception=ConfigException)
        smith.assert_exists(path, 'benchmarks', exception=ConfigException)
        smith.assert_exists(path, 'datasets', exception=ConfigException)
        smith.assert_exists(path, 'parboil', exception=ConfigException)
        return path

    return verify_parboil("~/src/parboil")


def clsmith_path():
    return smith.assert_exists("extern/clsmith", exception=ConfigException)


def clsmith():
    return smith.assert_exists(clsmith_path(), "build", "CLSmith",
                               exception=ConfigException)


def llvm_path():
    return smith.assert_exists(phd_root(), "tools", "llvm",
                               exception=ConfigException)


def libclc():
    return smith.assert_exists(phd_root(), "extern", "libclc",
                               exception=ConfigException)


def clang():
    return smith.assert_exists(llvm_path(), "build", "bin", "clang",
                               exception=ConfigException)


def rewriter():
    return smith.assert_exists(phd_root(), "src", "smith", "native", "rewriter",
                               exception=ConfigException)


def toolchain_env():
    return {'LD_LIBRARY_PATH': fs.path(llvm_path(), "build", "lib")}


def opt():
    return smith.assert_exists(llvm_path(), "build", "bin", "opt",
                               exception=ConfigException)


def is_host():
    """
    :return: True if host, False if device.
    """
    return gethostname() != "whz4"
