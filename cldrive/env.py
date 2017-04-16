import os
import platform
import re
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


def opencl_version(platform: cl.Platform) -> str:
    """
    Get supported OpenCL version of a platform.

    Parameters
    ----------
    platform : cl.Platform
        Pyopencl platform object.

    Returns
    -------
    str
        OpenCL version supported by platform.

    Raises
    ------
    TypeError:
        If parameter is of invalid type.
    LookupError:
        If the OpenCL version cannot be determined from the output of
        CL_PLATFORM_VERSION.

    Examples
    --------
    With a platform that supports OpenCL 1.2:
    >>> opencl_version(platform1)  # doctest: +SKIP
    "1.2"

    With a platform that supports OpenCL 2.0:
    >>> opencl_version(platform2)  # doctest: +SKIP
    "2.0"
    """
    if not isinstance(platform, cl.Platform):
        raise TypeError("not a pyopencl platform")

    version = platform.get_info(cl.platform_info.VERSION)
    m = re.match(r'OpenCL (\d+\.\d+)', version)
    if m:
        return m.group(1)
    else:
        raise LookupError(
            f"Could not determine OpenCL version from string '{version}'")


def device_type(device: cl.Device) -> str:
    """
    Get the type of an OpenCL device.

    Parameters
    ----------
    device : cl.Device
        Pyopencl device object.

    Returns
    -------
    str
        OpenCL device type.

    Raises
    ------
    TypeError:
        If parameter is of invalid type.

    Examples
    --------
    On a GPU device:
    >>> opencl_version(device1)  # doctest: +SKIP
    "GPU"

    On a CPU device:
    >>> opencl_version(device2)  # doctest: +SKIP
    "CPU"

    Using oclgrind:
    >>> opencl_version(device)  # doctest: +SKIP
    "Emulator"
    """
    cl_device_type = device.get_info(cl.device_info.TYPE)
    if cl_device_type == 15:
        # add special work-around for non-standard value '15', which is used
        # by oclgrind.
        return "Emulator"
    else:
        try:
            return cl.device_type.to_string(cl_device_type)
        except ValueError:
            return int(cl_device_type)


def clinfo(file=sys.stdout) -> None:
    print("Host:", host_os(), file=file)

    for platform_id, platform in enumerate(cl.get_platforms()):
        platform_name = platform.get_info(cl.platform_info.NAME)

        ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)])
        devices = ctx.get_info(cl.context_info.DEVICES)

        version = opencl_version(platform)

        print((f"Platform {platform_id}: "
               f"{platform_name} "
               f"(OpenCL {version})"), file=file)

        for device_id, device in enumerate(devices):
            device_name = device.get_info(cl.device_info.NAME)
            devtype = device_type(device)
            driver = device.get_info(cl.device_info.DRIVER_VERSION)

            print(f"    Device {device_id}: {device_name} "
                  f"(Type: {devtype}, Driver: {driver})", file=file)


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


def has_cpu() -> bool:
    """
    Determine if there is a CPU OpenCL device available.

    Returns
    -------
    bool
        True if device available, else False.
    """
    try:
        make_env(devtype="cpu")
    except LookupError:
        return False


def has_gpu() -> bool:
    """
    Determine if there is a CPU OpenCL device available.

    Returns
    -------
    bool
        True if device available, else False.
    """
    try:
        make_env(devtype="cpu")
    except LookupError:
        return False
