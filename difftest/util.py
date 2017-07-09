"""
Shared utility code for Jupyter notebooks.
"""
from collections import Counter

HOSTS = {
    "CentOS Linux 7.1.1503 64bit": "CentOS 7.1 64bit"
}

# shorthand device names
DEVICES = {
    "GeForce GTX 1080": "NVIDIA GTX 1080",
    "Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz": "Intel E5-2620 v4",
    "Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz": "Intel E5-2650 v2",
    "Olcgrind Simulator": "Oclgrind",
    "pthread-Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz": "Intel E5-2620 (pocl)",
    "Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz": "Intel i5-4570",
    'Intel(R) HD Graphics Haswell GT2 Desktop': 'Intel HD Haswell GT2',
    'Intel(R) Many Integrated Core Acceleration Card': 'Intel Xeon Phi (?)',
}

# shorthand driver names
DRIVERS = {
    "Oclgrind 16.10": "16.10",
}

# shorthand platform names
PLATFORMS = {
    "Intel(R) OpenCL": "Intel OpenCL",
    "Portable Computing Language": "POCL",
}

PLATFORMS_2_VENDORS = {
    "Intel(R) OpenCL": "intel",
    "Intel Gen OCL Driver": "intel",
    "Portable Computing Language": "pocl",
    "NVIDIA CUDA": "nvidia",
}

DEVTYPES = {
    "3": "CPU",
    "ACCELERATOR", "Accelerator",
}

# Ordering for the paper:
TESTBED_IDS = [3, 13, 9, 14, 10, 12, 11]
OCLGRIND_ID = 11
CONFIGURATIONS = list(zip(range(1, len(TESTBED_IDS) + 1), TESTBED_IDS))


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


def get_majority_output(session, result, table):
    results = session.query(table)\
        .filter(table.program == result.program,
                table.params == result.params,
                table.status == 0).all()

    if len(results) > 2:
        # Use voting to pick oracle.
        outputs = [r.stdout for r in results]
        majority_output, majority_count = Counter(outputs).most_common(1)[0]
    elif len(results) == 2:
        if results[0].stdout != results[1].stdout:
            majority_count = 1
            ndistinct = len(results)
            majority_output = f"[UNKNOWN] ({ndistinct} distinct outputs)"
        else:
            majority_count = 2

    majority_devices = [
        r.testbed for r in results if r.stdout == majority_output
    ]
    minority_devices = [
        r.testbed for r in results if r.stdout != majority_output
    ]
    minority_count = len(minority_devices)

    return majority_output, majority_devices
