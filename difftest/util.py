"""
Shared utility code for Jupyter notebooks.
"""
from collections import Counter, namedtuple
from labm8 import crypto
from progressbar import ETA, ProgressBar


from db import *


HOSTS = {
    "CentOS Linux 7.1.1503 64bit": "CentOS 7.1 64bit"
}

# shorthand device names
DEVICES = {
    "Codeplay Software Ltd. - host CPU": "ComputeAorta (Intel E5-2620)",
    "Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz": "Intel i5-4570",
    "Intel(R) HD Graphics Haswell GT2 Desktop": "Intel HD Haswell GT2",
    "Intel(R) Many Integrated Core Acceleration Card": "Intel Xeon Phi",
    "Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz": "Intel E5-2620 v4",
    "Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz": "Intel E5-2650 v2",
    "Olcgrind Simulator": "Oclgrind",
    "pthread-Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz": "POCL (Intel E5-2620)",
}

# shorthand driver names
DRIVERS = {
    "Oclgrind 16.10": "16.10",
}

# shorthand platform names
PLATFORMS = {
    "Intel Gen OCL Driver": "Beignet",
    "Intel(R) OpenCL": "Intel OpenCL",
    "Portable Computing Language": "POCL",
}

PLATFORMS_2_VENDORS = {
    "ComputeAorta": "codeplay",
    "Intel Gen OCL Driver": "intel",
    "Intel(R) OpenCL": "intel",
    "NVIDIA CUDA": "nvidia",
    "Oclgrind": "oclgrind",
    "Portable Computing Language": "pocl",
}


def vendor_str(platform):
    return PLATFORMS_2_VENDORS[platform]


DEVTYPES = {
    "3": "CPU",
    "ACCELERATOR": "Accelerator",
}

# Ordering for the paper:
TESTBED_IDS = [3, 20, 13, 9, 14, 10, 15, 12, 22, 11]
OCLGRIND_ID = 11
Configuration = namedtuple('Configuration', ['id', 'testbed_id'])

CONFIGURATIONS = [Configuration(*x) for x in zip(range(1, len(TESTBED_IDS) + 1), TESTBED_IDS)]


def platform_str(platform: str):
    platform = platform.strip()
    return PLATFORMS.get(platform, platform)


def device_str(device: str):
    device = device.strip()
    return DEVICES.get(device, device)


def driver_str(driver: str):
    driver = driver.strip()
    return DRIVERS.get(driver, driver)


def host_str(host: str):
    host = host.strip()
    return HOSTS.get(host, host)


def devtype_str(devtype: str):
    devtype = devtype.strip()
    return DEVTYPES.get(devtype, devtype)


def NamedProgressBar(name):
    """
    TODO: Return progress bar with named prefix.
    """
    # return ProgressBar(widgets=[f'{name} :: ', ETA()])
    return ProgressBar()


def escape_stdout(stdout):
    """ filter noise from test harness stdout """
    return '\n'.join(line for line in stdout.split('\n')
                     if line != "ADL Escape failed."
                     and line != "WARNING:endless loop detected!"
                     and line != "One module without kernel function!")


def escape_stderr(stderr):
    """ filter noise from test harness stderr """
    return '\n'.join(line for line in stderr.split('\n')
                     if "no version information available" not in line)


def get_assertion(s: session_t, table, stderr: str, clang_assertion: bool=True):
    for line in stderr.split('\n'):
        if "assertion" in line.lower():
            if clang_assertion:
                msg = ":".join(line.split(":")[3:])
            else:
                msg = line
            assertion = get_or_create(
                s, table,
                hash=crypto.sha1_str(msg),
                assertion=msg)
            s.add(assertion)
            s.flush()
            return assertion


def get_unreachable(s: session_t, tables, stderr: str):
    for line in stderr.split('\n'):
        if "unreachable executed at" in line.lower():
            unreachable = get_or_create(
                s, table,
                hash=crypto.sha1_str(line),
                unreachable=line)
            s.add(unreachable)
            s.flush()
            return unreachable


def get_terminate(s: session_t, tables, stderr: str):
    for line in stderr.split('\n'):
        if "terminate called after throwing an instance" in line.lower():
            terminate = get_or_create(
                s, table,
                hash=crypto.sha1_str(line),
                terminate=line)
            s.add(terminate)
            s.flush()
            return terminate
