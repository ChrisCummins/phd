import os
import re
import subprocess

from subprocess import Popen

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


def train_cmd(input_h5, input_json, rnn_size=1024, num_layers=3,
              model_type="lstm", seq_length=250, max_epochs=100,
              init_from=None):
    """
    Get torch_rnn command.

    :param input_h5: .
    :param input_json: .
    :param rnn_size: RNN size.
    :param num_layers: The number of layers.
    :param model_type: The model type.
    :param seq_length: The sequence length.
    :param max_epochs: The number of training epochs.
    :param init_from: The checkpoint path.
    """
    cmd = [
        'th', 'train.lua',
        '-input_h5', input_h5,
        '-input_json', input_json,
        '-reset_iterations', '0'
        '-rnn_size', rnn_size,
        '-num_layers', num_layers,
        '-model_type', model_type,
        '-seq_length', seq_length,
        '-print_every', 100,
        '-checkpoint_every', 1000,
        '-max_expochs', max_epochs
    ]
    if init_from:
        cmd += ['-init_from', init_from]
    return cmd


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


def train(input_h5, input_json, rnn_size=1024, num_layers=3, model_type="lstm",
          seq_length=250, max_epochs=100, max_num_attempts=1000):
    """
    Run torch-rnn train for the specified number of attempts.

    :param input_h5: .
    :param input_json: .
    :param rnn_size: RNN size.
    :param num_layers: The number of layers.
    :param model_type: The model type.
    :param seq_length: The sequence length.
    :param max_epochs: The number of training epochs.
    :param max_num_attempts: The maximum number of attempts.
    """
    cmd = train_cmd(input_h5, input_json, rnn_size=rnn_size,
                    num_layers=num_layers, model_type=model_type,
                    seq_length=seq_length, max_epochs=max_epochs)

    os.chdir(path())
    i = 0
    while i < max_num_attempts:
        print('  -> training attempt {}'.format(i))
        process = Popen(cmd)
        if process.returncode == 0:
            break
        else:
            i += 1
