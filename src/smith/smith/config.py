import smith

import os

from socket import gethostname

class ConfigException(smith.SmithException): pass


def torch_rnn_path():
    """
    Get path to Torch RNN.

    :return: Path to Torch RNN.
    """
    path = os.path.expanduser("~/src/torch-rnn")

    if not os.path.exists(path):
        raise ConfigException("Torch RNN root '{}' not found"
                              .format(path))
    return path


def parboil_root():
    """
    Get path to Parboil benchmarks.

    :return: Path to Parboil.
    """
    def verify_parboil(path):
        if not os.path.exists(path):
            raise ConfigException("Parboil root '{}' not found"
                                  .format(path))
        benchmarks = os.path.join(path, 'benchmarks')
        if not os.path.exists(benchmarks):
            raise ConfigException("Parboil benchmarks '{}' not found"
                                  .format(benchmarks))
        datasets = os.path.join(path, 'datasets')
        if not os.path.exists(datasets):
            raise ConfigException("Parboil datasets '{}' not found"
                                  .format(datasets))
        driver = os.path.join(path, 'parboil')
        if not os.path.exists(driver):
            raise ConfigException("Parboil driver '{}' not found"
                                  .format(driver))
        return path

    path = os.path.abspath(os.path.expanduser("~/src/parboil"))
    return verify_parboil(path)


def is_host():
    """
    :return: True if host, False if device.
    """
    return gethostname() != "whz4"
