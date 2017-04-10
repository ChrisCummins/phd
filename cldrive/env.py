import os
import platform
import sys

from collections import namedtuple
from typing import List, Tuple

import pyopencl as cl


class OpenCLEnvironment(namedtuple('OpenCLEnvironment', ['platform', 'device'])):
    __slots__ = ()  # memory saving

    def __repr__(self) -> str:
        return f"Device: {self.device}, Platform: {self.platform}"

    def ctx_queue(self, profiling=False) -> Tuple[cl.Context, cl.CommandQueue]:
        """
        Return an OpenCL context and command queue for the given environment.

        Arguments
        ---------
        profiling : bool, optional
            If True, enable profiling support.

        Raises
        ------
        LookupError
            If a matching OpenCL device cannot be found.
        RuntimeError
            In case of an OpenCL API call failure.
        """
        return _lookup_env(return_cl=True, platform=self.platform,
                           device=self.device, profiling=profiling)


def _cl_devtype_from_str(string: str) -> cl.device_type:
    devtypes = {
        "cpu": cl.device_type.CPU,
        "gpu": cl.device_type.GPU,
        "all": cl.device_type.ALL
    }

    try:
        return devtypes[string.lower()]
    except KeyError:
        raise ValueError(f"unrecognized device type '{string}'")


def _devtype_matches(device: cl.Device, devtype: cl.device_type) -> bool:
    """ check that device type matches """
    if devtype == cl.device_type.ALL:
        return True
    else:
        actual_devtype = device.get_info(cl.device_info.TYPE)
        return actual_devtype == devtype


def _lookup_env(return_cl: bool, platform: str=None, device: str=None,
                devtype: str="all", profiling: bool=False) -> OpenCLEnvironment:
    """ find a matching OpenCL device """
    cl_devtype = _cl_devtype_from_str(devtype)

    try:
        cl_platforms = cl.get_platforms()
        if not len(cl_platforms):
            raise LookupError("no OpenCL platforms available")

        for cl_platform in cl_platforms:
            platform_str = cl_platform.get_info(cl.platform_info.NAME)

            if platform and platform != platform_str:
                continue

            ctx = cl.Context(
                properties=[(cl.context_properties.PLATFORM, cl_platform)])

            cl_devices = ctx.get_info(cl.context_info.DEVICES)

            # filter devices on device type
            cl_devices = [d for d in cl_devices if _devtype_matches(d, cl_devtype)]

            for cl_device in cl_devices:
                device_str = cl_device.get_info(cl.device_info.NAME)

                if device and device != device_str:
                    continue

                if return_cl:
                    if profiling:
                        properties = cl.command_queue_properties.PROFILING_ENABLE
                    else:
                        properties = None

                    queue = cl.CommandQueue(ctx, device=cl_device,
                                            properties=properties)
                    return ctx, queue
                else:
                    return OpenCLEnvironment(
                        platform=platform_str, device=device_str)
            else:
                if device:
                    raise LookupError(
                        f"could not find device '{device}' on platform '{platform}'")
        else:
            if platform:
                raise LookupError(
                        f"could not find platform '{platform}'")

        raise LookupError(f"could not find a device of type '{devtype}'")
    except cl.RuntimeError as e:
        raise RuntimeError from e


def host_os() -> str:
    """
    Get the type and version of the host operating system.

    Returns
    -------
    str
        Formatted <system> <release> <arch>, where <system> is the
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


def make_env(platform: str=None, device: str=None,
             devtype: str="all") -> OpenCLEnvironment:
    """
    Create an OpenCL context and device queue.

    Iterates over the available OpenCL platforms and devices looking for a
    device matching the requested platform ID, or platform and device ID, or
    device type. Constructs and returns an OpenCL context and queue for the
    matching device. Note that OpenCL profiling is enabled.

    Parameters
    ----------
    platform_id : int, optional
        OpenCL Platform ID. If not provided, any available platform may be
        used.
    device_id : int, optional
        OpenCL Device ID. If not provided, any available device may be used.
        Requires that platform_id is set.
    devtype : str, optional
        OpenCL device type to use, one of: {all,cpu,gpu}.

    Returns
    -------
    OpenCLEnvironment
        A named tuple consisting of the platform and device name.

    Raises
    ------
    ValueError
        If device_id is set, but not platform_id.
    LookupError
        If no matching device found.
    RuntimeError
        In case OpenCL API call fails.
    """
    return _lookup_env(return_cl=False, platform=platform, device=device,
                       devtype=devtype)
