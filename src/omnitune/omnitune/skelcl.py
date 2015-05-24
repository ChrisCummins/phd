import re
import time

import dbus
import dbus.service
import dbus.mainloop.glib
import gobject

import labm8
from labm8 import crypto
from labm8 import fs
from labm8 import io

import omnitune
from omnitune import cache
from omnitune import util
from omnitune import llvm
from omnitune import opencl


SESSION_NAME   = "org.omnitune"
INTERFACE_NAME = "org.omnitune.skelcl"
OBJECT_PATH    = "/"


STENCIL_KERNEL_FEATURES = (
    "instructions (of all types)",
    "ratio AShr insts",
    "ratio Add insts",
    "ratio Alloca insts",
    "ratio And insts",
    "ratio Br insts",
    "ratio Call insts",
    "ratio FAdd insts",
    "ratio FCmp insts",
    "ratio FDiv insts",
    "ratio FMul insts",
    "ratio FPExt insts",
    "ratio FPToSI insts",
    "ratio FSub insts",
    "ratio GetElementPtr insts",
    "ratio ICmp insts",
    "ratio InsertValue insts",
    "ratio Load insts",
    "ratio Mul insts",
    "ratio Or insts",
    "ratio PHI insts",
    "ratio Ret insts",
    "ratio SDiv insts",
    "ratio SExt insts",
    "ratio SIToFP insts",
    "ratio SRem insts",
    "ratio Select insts",
    "ratio Shl insts",
    "ratio Store insts",
    "ratio Sub insts",
    "ratio Trunc insts",
    "ratio UDiv insts",
    "ratio Xor insts",
    "ratio ZExt insts",
    "ratio basic blocks",
    "ratio memory instructions",
    "ratio non-external functions"
)


OPENCL_DEVICE_FEATURES = (
    "address_bits",
    "double_fp_config",
    "endian_little",
    "execution_capabilities",
    "extensions",
    "global_mem_cache_size",
    "global_mem_cache_type",
    "global_mem_cacheline_size",
    "global_mem_size",
    "host_unified_memory",
    "image2d_max_height",
    "image2d_max_width",
    "image3d_max_depth",
    "image3d_max_height",
    "image3d_max_width",
    "image_support",
    "local_mem_size",
    "local_mem_type",
    "max_clock_frequency",
    "max_compute_units",
    "max_constant_args",
    "max_constant_buffer_size",
    "max_mem_alloc_size",
    "max_parameter_size",
    "max_read_image_args",
    "max_samplers",
    "max_work_group_size",
    "max_work_item_dimensions",
    "max_work_item_sizes[0]",
    "max_work_item_sizes[1]",
    "max_work_item_sizes[2]",
    "max_write_image_args",
    "mem_base_addr_align",
    "min_data_type_align_size",
    "name",
    "native_vector_width_char",
    "native_vector_width_double",
    "native_vector_width_float",
    "native_vector_width_half",
    "native_vector_width_int",
    "native_vector_width_long",
    "native_vector_width_short",
    "preferred_vector_width_char",
    "preferred_vector_width_double",
    "preferred_vector_width_float",
    "preferred_vector_width_half",
    "preferred_vector_width_int",
    "preferred_vector_width_long",
    "preferred_vector_width_short",
    "queue_properties",
    "single_fp_config",
    "type",
    "vendor",
    "vendor_id",
    "version"
)


class Error(Exception):
    """
    Module-level base error class.
    """
    pass


class FeatureExtractionError(Error):
    """
    Error thrown if feature extraction fails.
    """
    pass


def checksum_str(string):
    """
    Return the checksum for a string.
    """
    return crypto.sha1(string)


def vectorise_ratios(ratios):
    """
    Vectorise a dictionary of stencil kernel features.
    """
    vector = []
    for feature in STENCIL_KERNEL_FEATURES:
        if feature in ratios:
            vector.append(ratios[feature])
        else:
            vector.append(0)
    return vector


def vectorise_devinfo(info):
    """
    Vectorise a dictionary of OpenCL device info.
    """
    vector = []
    for feature in OPENCL_DEVICE_FEATURES:
        vector.append(info[feature])

    return vector


def get_user_source(source):
    """
    Return the user source code for a stencil kernel.

    This strips the common stencil implementation, i.e. the border
    loading logic.

    Raises:
        FeatureExtractionError if the "end of user code" marker is not found.
    """
    lines = source.split("\n")
    user_source = []
    for line in lines:
        if line == "// --- SKELCL END USER CODE ---":
            return "\n".join(user_source)
        user_source.append(line)

    raise FeatureExtractionError("Failed to find end of user code marker")


def get_source_features(source, path=""):
    user_source = get_user_source(source)
    bitcode = llvm.bitcode(user_source, path=path)
    instcounts = llvm.instcounts(bitcode, path=path)
    ratios = llvm.instcounts2ratios(instcounts)

    return vectorise_ratios(ratios)


def get_local_device_features():
    devices = {}
    for info in opencl.get_devinfos():
        devices[info["name"]] = vectorise_devinfo(info)
    return devices


class SkelCLProxy(omnitune.Proxy):

    LLVM_PATH = fs.path("~/src/msc-thesis/skelcl/libraries/llvm/build/bin/")

    def __init__(self, *args, **kwargs):
        """
        Construct a SkelCL proxy.
        """
        # Fail if we can't find the path
        if not fs.isdir(self.LLVM_PATH):
            io.fatal("Could not find llvm path '{0}'".format(self.LLVM_PATH))

        super(SkelCLProxy, self).__init__(*args, **kwargs)
        io.info("Registered proxy %s/SkelCLProxy ..." % SESSION_NAME)
        self.kcache = cache.JsonCache("/tmp/omnitune-skelcl-kcache.json")
        self.dcache = cache.JsonCache("/tmp/omnitune-skelcl-dcache.json")

        # Add local device features to dcache.
        for device,info in get_local_device_features().iteritems():
            io.debug("Local device: '{0}'".format(device))
            self.dcache.set(device, info)

    @dbus.service.method(INTERFACE_NAME, in_signature='siiiiiiis', out_signature='(nn)')
    def RequestStencilParams(self, device_name, device_count,
                             north, south, east, west, data_width,
                             data_height, source):
        """
        Request parameter values for a SkelCL stencil operation.

        Determines the parameter values to use for a SkelCL stencil
        operation, using a machine learning classifier to predict the
        optimal parameter values given a set of features determined
        from the arguments.

        Args:
            device_name: The name of the execution device, as returned by
                OpenCL getDeviceInfo() API.
            device_count: The number of execution devices.
            north: The stencil shape north direction.
            south: The stencil shape south direction.
            east: The stencil shape east direction.
            west: The stencil shape west direction.
            data_width: The number of columns of input data.
            data_height: The number of rows of input data.
            source: The stencil kernel source code.

        Returns:
            A tuple of work group size values, e.g.

            (16,32)
        """

        start_time = time.time()

        # Parse arguments.
        device_name = util.parse_str(device_name)
        device_count = int(device_count)
        north = int(north)
        south = int(south)
        east = int(east)
        west = int(west)
        data_width = int(data_width)
        data_height = int(data_height)
        source = util.parse_str(source)

        # Calculate checksum of source code.
        checksum = checksum_str(source)

        # Get the source features.
        sourcefeatures = self.kcache.get(checksum)
        if sourcefeatures is None:
            features = get_source_features(source, path=self.LLVM_PATH)
            sourcefeatures = self.kcache.set(checksum, features)

        # Get the device features.
        devicefeatures = self.dcache.get(device_name)
        if devicefeatures is None:
            raise FeatureExtractionError(("Failed to lookup device features for "
                                          "'{0}'".format(device_name)))

        # Assemble the full features vector.
        features = devicefeatures + sourcefeatures + [
            north, south, east, west,
            data_width, data_height,
            device_count,
        ]

        wg = (64, 32)

        end_time = time.time()

        io.debug(("RequestStencilParams({dev}, {count}, "
                  "[{n}, {s}, {e}, {w}], {width}, {height}, {id}) -> "
                  "({c}, {r}) [{t:.3f}s]"
                  .format(dev=device_name.strip()[:8],
                          count=device_count,
                          n=north, s=south, e=east, w=west,
                          width=data_width, height=data_height,
                          id=checksum[:8],
                          c=wg[0], r=wg[1], t=end_time - start_time)))

        return wg

    @dbus.service.method(INTERFACE_NAME, in_signature='', out_signature='')
    def Exit(self):
        mainloop.quit()


def main():
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    bus = dbus.SystemBus()
    name = dbus.service.BusName(SESSION_NAME, bus)
    io.info("Launched session %s ..." % SESSION_NAME)

    # Launch SkelCL proxy.
    SkelCLProxy(bus, OBJECT_PATH)

    mainloop = gobject.MainLoop()
    try:
        mainloop.run()
    except KeyboardInterrupt:
        labm8.exit()
