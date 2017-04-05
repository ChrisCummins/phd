import platform
import sys

from typing import List

import pyopencl as cl


def os() -> str:
    """
    Get the type and version of the host operating system.

    Returns:
        str: Formatted <system> <release> <arch>, where <system> is the
            operating system type, <release> is the release version, and
            <arch> is 32 or 64 bit.
    """
    if sys.platform == "linux" or sys.platform == "linux2":
        dist = platform.linux_distribution()
        system, release = dist[0], dist[1]
    else:
        system, release = platform.system(), platform.release()

    arch = platform.architecture()[0]

    return f"{system} {release} {arch}"


def platform_name(platform_id: int) -> str:
    """
    Get the OpenCL platform name.

    Arguments:
        platform_id (int): ID of platform.

    Returns:
        str: Platform name.
    """
    platform = cl.get_platforms()[platform_id]
    return platform.get_info(cl.platform_info.NAME)


def device_name(platform_id: int, device_id: int) -> str:
    """
    Get the OpenCL device name.

    Arguments:
        platform_id (int): ID of platform.
        device_id (int): ID of device.

    Returns:
        str: Device name.
    """
    platform = cl.get_platforms()[platform_id]
    ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)])
    device = ctx.get_info(cl.context_info.DEVICES)[device_id]
    return device.get_info(cl.device_info.NAME)


def driver_version(platform_id: int, device_id: int) -> str:
    """
    Get the OpenCL device driver version.

    Arguments:
        platform_id (int): ID of platform.
        device_id (int): ID of device.

    Returns:
        str: Driver version string.
    """
    platform = cl.get_platforms()[platform_id]
    ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)])
    device = ctx.get_info(cl.context_info.DEVICES)[device_id]
    return device.get_info(cl.device_info.DRIVER_VERSION)


def print_summary(file=sys.stdout) -> None:
    """
    Print a summary of OpenCL platforms and device.

    Arguments:
        file (stream, optional): Output text stream.
    """
    print("Host:", os(), file=file)

    for platform_id, platform in enumerate(cl.get_platforms()):
        platform_name = platform.get_info(cl.platform_info.NAME)

        ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)])
        devices = ctx.get_info(cl.context_info.DEVICES)

        print(f"Platform {platform_id}: {platform_name}", file=file)

        for device_id, device in enumerate(devices):
            device_name = device.get_info(cl.device_info.NAME)
            device_type = cl.device_type.to_string(
                device.get_info(cl.device_info.TYPE))
            driver = device.get_info(cl.device_info.DRIVER_VERSION)

            print(f"    Device {device_id}: {device_type} {device_name} "
                  "{driver}", file=file)
