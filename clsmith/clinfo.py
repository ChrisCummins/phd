#!/usr/bin/env python3

import pyopencl as cl


def get_platform_name(platform_id):
    platform = cl.get_platforms()[platform_id]
    return platform.get_info(cl.platform_info.NAME)

def get_device_name(platform_id, device_id):
    platform = cl.get_platforms()[platform_id]
    ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)])
    device = ctx.get_info(cl.context_info.DEVICES)[device_id]
    return device.get_info(cl.device_info.NAME)

if __name__ == "__main__":
    for platform_id, platform in enumerate(cl.get_platforms()):
        platform_name = platform.get_info(cl.platform_info.NAME)

        ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)])
        devices = ctx.get_info(cl.context_info.DEVICES)

        print("Platform {platform_id}: {platform_name}".format(**vars()))

        for device_id, device in enumerate(devices):
            device_name = device.get_info(cl.device_info.NAME)
            device_type = cl.device_type.to_string(
                device.get_info(cl.device_info.TYPE))

            print("    Device {device_id}: {device_type} {device_name}"
                  .format(**vars()))
        print()
