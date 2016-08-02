import smith

import os

from socket import gethostname

import labm8
from labm8 import fs

class ConfigException(smith.SmithException): pass


def assert_exists(*path_components):
    path = fs.path(*path_components)
    if not os.path.exists(path):
        raise ConfigException("file '{}' not found".format(path))
    return path


def torch_rnn_path():
    """
    Get path to Torch RNN.

    :return: Path to Torch RNN.
    """
    return assert_exists("~/src/torch-rnn")


def parboil_root():
    """
    Get path to Parboil benchmarks.

    :return: Path to Parboil.
    """
    def verify_parboil(path):
        assert_exists(path)
        assert_exists(path, 'benchmarks')
        assert_exists(path, 'datasets')
        assert_exists(path, 'parboil')
        return path

    return verify_parboil("~/src/parboil")


def is_host():
    """
    :return: True if host, False if device.
    """
    return gethostname() != "whz4"
