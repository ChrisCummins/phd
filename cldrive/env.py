import os
import platform
import re
import sys

from collections import namedtuple
from typing import List, Tuple, Iterator

import pyopencl as cl


class OpenCLEnvironment(namedtuple('OpenCLEnvironment', ['platform', 'device'])):
    __slots__ = ()  # memory saving

    def __repr__(self) -> str:
        return f"Device: {self.device}, Platform: {self.platform}"

    def ctx_queue(self, profiling=False) -> Tuple[cl.Context, cl.CommandQueue]:
        """
        Return an OpenCL context and command queue for the given environment.

        Parameters
        ----------
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

    def ids(self) -> Tuple[int, int]:
        """
        Return platform and device ID numbers.

        The ID numbers can be used to index into the list of platforms and
        devices. Note that the stability of these IDs is *not* guaranteed
        by OpenCL, and may depend on ICD load order or any number of other
        factors.

        Raises:
        -------
        LookupError
            If a matching OpenCL device cannot be found.
        RuntimeError
            In case of an OpenCL API call failure.
        """
        return _lookup_env(return_ids=True, platform=self.platform,
                           device=self.device)

    @property
    def driver_version(self) -> str:
        """
        Get the OpenCL driver version.

        Returns
        -------
        str
            OpenCL device version.

        Raises
        ------
        LookupError
            If a matching OpenCL device cannot be found.
        RuntimeError
            In case of an OpenCL API call failure.

        Examples
        --------
        make_env().driver_version  # doctest: +SKIP
        "375.39"
        """
        ctx, queue = self.ctx_queue()
        dev = queue.get_info(cl.command_queue_info.DEVICE)
        return dev.get_info(cl.device_info.DRIVER_VERSION)

    @property
    def opencl_version(self) -> str:
        """
        Get the OpenCL platform version.

        Returns
        -------
        str
            OpenCL platform version.

        Raises
        ------
        LookupError
            If a matching OpenCL device cannot be found.
        RuntimeError
            In case of an OpenCL API call failure.

        Examples
        --------
        make_env().platform_version  # doctest: +SKIP
        "1.2"
        """
        ctx, queue = self.ctx_queue()
        dev = queue.get_info(cl.command_queue_info.DEVICE)
        plat = dev.get_info(cl.device_info.PLATFORM)
        return opencl_version(plat)

    @property
    def device_type(self) -> str:
        """
        Get the OpenCL device type.

        Returns
        -------
        str
            OpenCL device type.

        Raises
        ------
        LookupError
            If a matching OpenCL device cannot be found.
        RuntimeError
            In case of an OpenCL API call failure.

        Examples
        --------
        make_env().device_type  # doctest: +SKIP
        "CPU"
        """
        ctx, queue = self.ctx_queue()
        dev = queue.get_info(cl.command_queue_info.DEVICE)
        return device_type(dev)


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


def _lookup_env(return_cl: bool=False, return_ids: bool=False, platform: str=None,
                device: str=None, devtype: str="all", profiling: bool=False):
    """ find a matching OpenCL device """
    cl_devtype = _cl_devtype_from_str(devtype)

    try:
        cl_platforms = cl.get_platforms()
        if not len(cl_platforms):
            raise LookupError("no OpenCL platforms available")

        for platform_id, cl_platform in enumerate(cl_platforms):
            platform_str = cl_platform.get_info(cl.platform_info.NAME)

            if platform and platform != platform_str:
                continue

            ctx = cl.Context(
                properties=[(cl.context_properties.PLATFORM, cl_platform)])

            cl_devices = ctx.get_info(cl.context_info.DEVICES)

            # filter devices on device type
            cl_devices = [d for d in cl_devices if _devtype_matches(d, cl_devtype)]

            for device_id, cl_device in enumerate(cl_devices):
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
                elif return_ids:
                    return platform_id, device_id
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
    """
    Print a summary of available OpenCL devices.
    """
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
    platform : str, optional
        OpenCL Platform name. If not provided, any available platform may be
        used.
    device : str, optional
        OpenCL Device name. If not provided, any available device may be used.
    devtype : str, optional
        OpenCL device type to use, one of: {all,cpu,gpu}.

    Returns
    -------
    OpenCLEnvironment
        A named tuple consisting of the platform and device name.

    Raises
    ------
    LookupError
        If no matching device found.
    RuntimeError
        In case an OpenCL API call fails.

    Examples
    --------
    To generate an environment for the first available device:
    >>> make_env()  # doctest: +SKIP
    Device: ..., Platform: ...

    To generate an environment for a GPU device:
    >>> make_env(devtype="gpu")  # doctest: +SKIP
    Device: ..., Platform: ...

    To generate an environment for a specific device:
    >>> make_env(platform="NVIDIA CUDA", device="GeForce GTX 1080")  # doctest: +SKIP
    Device: GeForce GTX 1080, Platform: NVIDIA CUDA
    """
    return _lookup_env(return_cl=False, platform=platform, device=device,
                       devtype=devtype)


def all_envs(devtype: str='all') -> Iterator[OpenCLEnvironment]:
    """
    Iterate over all available OpenCL environments on a system.

    Parameters
    ----------
    devtype : str, optional
        OpenCL device type to filter by, one of: {all,cpu,gpu}.

    Returns
    -------
    Iterator[OpenCLEnvironment]
        An iterator over all available OpenCL environments.
    """
    cl_devtype = _cl_devtype_from_str(devtype)
    cl_platforms = cl.get_platforms()
    for cl_platform in cl_platforms:
        platform_str = cl_platform.get_info(cl.platform_info.NAME)
        ctx = cl.Context(
                properties=[(cl.context_properties.PLATFORM, cl_platform)])

        cl_devices = ctx.get_info(cl.context_info.DEVICES)
        cl_devices = [d for d in cl_devices if _devtype_matches(d, cl_devtype)]

        for cl_device in cl_devices:
            device_str = cl_device.get_info(cl.device_info.NAME)

            yield OpenCLEnvironment(platform=platform_str, device=device_str)


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
        return True
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
        return True
    except LookupError:
        return False
