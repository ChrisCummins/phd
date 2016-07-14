import os
import re
import subprocess

import smith
from smith import config


class TorchRnnException(smith.SmithException): pass


def path():
    """
    Get path to Torch RNN.

    :return: Path to Torch RNN.
    """
    try:
        return config.torch_rnn_path()
    except config.ConfigException as e:
        raise TorchRnnException(str(e))


def fetch_samples(sample_path):
    """
    Fetch OpenCL samples.

    :param sample_path: Path to sample file.
    :return: List of strings.
    """
    with open(sample_path) as infile:
        contents = infile.read()
        samples = re.split(r'=== SAMPLE [0-9]+ ===', contents)
    return [sample.strip() for sample in samples if sample.strip()]


def checkpoint_path(path):
    """
    """
    path = os.path.expanduser(path)
    checkpoints = [os.path.join(path, x) for x in os.listdir(path)
                   if x.endswith('.t7')]

    if not checkpoints:
        raise TorchRnnException("No checkpoints found in '{}'".format(path))

    checkpoints.sort(key=os.path.getmtime)
    return checkpoints[-1]


def opencl_sample_command(checkpoint, seed, dest='/tmp/samples.txt',
                          temperature=0.75, length=5000,
                          num_samples=1000):
    """
    Build OpenCL sample command.

    :param checkpoint: Path to checkpoint file.
    :param seed: Seed string.
    :param dest: Path to destination file.
    :param temperature: Sample temperature.
    :param length: Sample length.
    :param num_samples: Number of samples to generate.
    :return: Sample command.
    """
    return ('th sample.lua -checkpoint "{checkpoint}" '
            '-temperature {temperature} -length {length} '
            '-opencl 1 -start_text "{seed}" -n {num_samples} &> {dest}'
            .format(checkpoint=checkpoint, temperature=temperature,
                    length=length, seed=seed, num_samples=num_samples,
                    dest=dest))


def sanitise_seed(seed):
    """
    Sanitise OpenCL seed. For torch-rnn, this means removing any line
    breaks.

    :param seed: Seed string.
    :return: Sanitised seed.
    """
    return re.sub('\n +', ' ', seed)


def opencl_samples(seed, num_samples=1000):
    """
    Generate OpenCL samples from a seed.

    Note this has the side effect of changing the working directory.

    :param seed: Seed.
    :return: List of samples as strings.
    """
    checkpoint_dir = os.path.join(path(), 'cv')
    checkpoint = checkpoint_path(checkpoint_dir)
    seed = sanitise_seed(seed)
    temperature = .75
    length = 5000
    cmd = opencl_sample_command(checkpoint, seed, temperature=temperature,
                                length=length, num_samples=num_samples)

    os.chdir(path())
    print('\r\033[K  -> seed: {}'.format(seed, end='')
    subprocess.call(cmd, shell=True)

    samples = fetch_samples('/tmp/samples.txt')

    print('\r\033[K', end='')
    return samples
