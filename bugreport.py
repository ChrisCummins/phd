#!/usr/bin/env python3

import psutil
import subprocess
import platform

import cldrive


def get_cpu():
    with open('/proc/cpuinfo') as infile:
        for line in infile.readlines():
            if line.startswith('model name'):
                return line.split(':')[1].strip()


def get_ram():
    ram = psutil.virtual_memory().total / (1024 ** 2)
    return f"{ram:.0f} MB"


def get_computer_name():
    dmi = subprocess.check_output('sudo dmidecode | grep "Product Name" | head -n1', shell=True)
    return dmi.decode('utf-8').split(':')[1].strip()


def get_os_kernel():
    return " ".join([platform.system(), platform.release(), platform.version()])


def main():
    os = cldrive.host_os()
    print(f"OS:  {os}")

    kernel = get_os_kernel()
    print(f"OS kernel:  {kernel}")

    cpu = get_cpu()
    print(f"CPU make and model:  {cpu}")

    ram = get_ram()
    print(f"Amount of system memory:  {ram}")

    pc = get_computer_name()
    print(f"PC make and model:  {pc}")


if __name__ == "__main__":
    main()
