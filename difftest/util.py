"""
Shared utility code for Jupyter notebooks.
"""
from collections import Counter


# shorthand device names
DEVICES = {
    "GeForce GTX 1080": "NVIDIA GTX 1080",
    "Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz": "Intel E5-2620",
    "Olcgrind Simulator": "Oclgrind",
    "pthread-Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz": "Intel E5-2620 (pocl)",
    "Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz": "Intel i5-4570",
    'Intel(R) HD Graphics Haswell GT2 Desktop': 'Intel HD Haswell GT2',
}

# shorthand driver names
DRIVERS = {
    "Oclgrind 16.10": "16.10",
}

# shorthand platform names
PLATFORMS = {
    "Portable Computing Language": "POCL",
}

PLATFORMS_2_VENDORS = {
    "Intel(R) OpenCL": "intel",
    "Intel Gen OCL Driver": "intel",
    "Portable Computing Language": "pocl",
    "NVIDIA CUDA": "nvidia"
}


def device_str(device):
    return DEVICES.get(device, device)


def driver_str(driver):
    return DRIVERS.get(driver, driver)


def platform_str(platform):
    return PLATFORMS.get(platform, platform)


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
