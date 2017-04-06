import os
import platform
import sys

from collections import namedtuple
from typing import List

import pyopencl as cl


OpenCLEnvironment = namedtuple('OpenCLEnvironment', ['ctx', 'queue'])


def host_os() -> str:
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


def clinfo(file=sys.stdout) -> None:
    print("Host:", host_os(), file=file)

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
                  f"{driver}", file=file)


def make_env(platform_id: int=None, device_id: int=None,
             devtype: str="all", queue_flags: int=0) -> OpenCLEnvironment:
    """
    Create an OpenCL context and device queue.

    Iterates over the available OpenCL platforms and devices looking for a
    device matching the requested platform ID, or platform and device ID, or
    device type. Constructs and returns an OpenCL context and queue for the
    matching device. Note that OpenCL profiling is enabled.

    Arguments:
        platform_id (int, optional): OpenCL Platform ID. If not provided, any
            available platform may be used.
        device_id (int, optional): OpenCL Device ID. If not provided, any
            available device may be used. Requires that platform_id is set.
        devtype (str, optional): OpenCL device type to use, one of:
            {all,cpu,gpu}.
        queue_flags (cl.command_queue_properties, optional): Bitfield of
            OpenCL queue constructor options.

    Returns:
        OpenCLEnvironment: A named tuple consisting of an OpenCL context and
            device queue.

    Raises:
        ValueError: If device_id is set, but not platform_id.
        LookupError: If no matching device found.
    """
    def device_type_matches(device: cl.Device,
                            cl_devtype: cl.device_type) -> bool:
        """ check that device type matches """
        if cl_devtype == cl.device_type.ALL:
            return True
        else:
            actual_devtype = device.get_info(cl.device_info.TYPE)
            return actual_devtype == cl_devtype

    if devtype == "cpu":
        cl_devtype = cl.device_type.CPU
    elif devtype == "gpu":
        cl_devtype = cl.device_type.GPU
    elif devtype == "all":
        cl_devtype = cl.device_type.ALL
    else:
        raise ValueError(f"unsupported device type '{devtype}'")

    # get list of platforms to iterate over. If platform ID is provided, use
    # only that platform.
    if platform_id is None:
        platforms = cl.get_platforms()
    else:
        try:
            platforms = [cl.get_platforms()[platform_id]]
        except IndexError:
            raise LookupError(f"No platform for id={platform_id}")

    for platform in platforms:
        ctx = cl.Context(
            properties=[(cl.context_properties.PLATFORM, platform)])

        # get list of devices to iterate over. If device ID is provided, use
        # only that device. Else, take any device which matches devtype
        if device_id is None:
            devices = ctx.get_info(cl.context_info.DEVICES)
        else:
            _assert_or_raise(platform_id is not None, ValueError)
            try:
                devices = [ctx.get_info(cl.context_info.DEVICES)[device_id]]
            except IndexError:
                raise LookupError(f"No device for id={device_id}")

        devices = [d for d in devices if device_type_matches(d, cl_devtype)]

        if len(devices):
            queue = cl.CommandQueue(ctx, device=devices[0],
                                    properties=queue_flags)
            return OpenCLEnvironment(ctx=ctx, queue=queue)

    raise LookupError("Could not find a suitable device")
