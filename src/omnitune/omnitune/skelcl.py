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
from labm8 import system

import omnitune
from omnitune import cache
from omnitune import db
from omnitune import util
from omnitune import llvm
from omnitune import opencl


SESSION_NAME   = "org.omnitune"
INTERFACE_NAME = "org.omnitune.skelcl"
OBJECT_PATH    = "/"


KERNEL_TABLE_SCHEMA = (
    ("checksum",                     "TEXT", "PRIMARY KEY"),
    ("source",                       "TEXT"),
    ("instruction_count",            "INTEGER"),
    ("ratio_AShr_insts",             "REAL"),
    ("ratio_Add_insts",              "REAL"),
    ("ratio_Alloca_insts",           "REAL"),
    ("ratio_And_insts",              "REAL"),
    ("ratio_Br_insts",               "REAL"),
    ("ratio_Call_insts",             "REAL"),
    ("ratio_FAdd_insts",             "REAL"),
    ("ratio_FCmp_insts",             "REAL"),
    ("ratio_FDiv_insts",             "REAL"),
    ("ratio_FMul_insts",             "REAL"),
    ("ratio_FPExt_insts",            "REAL"),
    ("ratio_FPToSI_insts",           "REAL"),
    ("ratio_FSub_insts",             "REAL"),
    ("ratio_GetElementPtr_insts",    "REAL"),
    ("ratio_ICmp_insts",             "REAL"),
    ("ratio_InsertValue_insts",      "REAL"),
    ("ratio_Load_insts",             "REAL"),
    ("ratio_Mul_insts",              "REAL"),
    ("ratio_Or_insts",               "REAL"),
    ("ratio_PHI_insts",              "REAL"),
    ("ratio_Ret_insts",              "REAL"),
    ("ratio_SDiv_insts",             "REAL"),
    ("ratio_SExt_insts",             "REAL"),
    ("ratio_SIToFP_insts",           "REAL"),
    ("ratio_SRem_insts",             "REAL"),
    ("ratio_Select_insts",           "REAL"),
    ("ratio_Shl_insts",              "REAL"),
    ("ratio_Store_insts",            "REAL"),
    ("ratio_Sub_insts",              "REAL"),
    ("ratio_Trunc_insts",            "REAL"),
    ("ratio_UDiv_insts",             "REAL"),
    ("ratio_Xor_insts",              "REAL"),
    ("ratio_ZExt_insts",             "REAL"),
    ("ratio_basic_blocks",           "REAL"),
    ("ratio_memory_instructions",    "REAL"),
    ("ratio_non_external_function",  "REAL")
)


DEVICE_TABLE_SCHEMA = (
    ("host",                           "TEXT"),
    ("address_bits",                   "INTEGER"),
    ("double_fp_config",               "INTEGER"),
    ("endian_little",                  "INTEGER"),
    ("execution_capabilities",         "INTEGER"),
    ("extensions",                     "TEXT"),
    ("global_mem_cache_size",          "INTEGER"),
    ("global_mem_cache_type",          "INTEGER"),
    ("global_mem_cacheline_size",      "INTEGER"),
    ("global_mem_size",                "INTEGER"),
    ("host_unified_memory",            "INTEGER"),
    ("image2d_max_height",             "INTEGER"),
    ("image2d_max_width",              "INTEGER"),
    ("image3d_max_depth",              "INTEGER"),
    ("image3d_max_height",             "INTEGER"),
    ("image3d_max_width",              "INTEGER"),
    ("image_support",                  "INTEGER"),
    ("local_mem_size",                 "INTEGER"),
    ("local_mem_type",                 "INTEGER"),
    ("max_clock_frequency",            "INTEGER"),
    ("max_compute_units",              "INTEGER"),
    ("max_constant_args",              "INTEGER"),
    ("max_constant_buffer_size",       "INTEGER"),
    ("max_mem_alloc_size",             "INTEGER"),
    ("max_parameter_size",             "INTEGER"),
    ("max_read_image_args",            "INTEGER"),
    ("max_samplers",                   "INTEGER"),
    ("max_work_group_size",            "INTEGER"),
    ("max_work_item_dimensions",       "INTEGER"),
    ("max_work_item_sizes_0",          "INTEGER"),
    ("max_work_item_sizes_1",          "INTEGER"),
    ("max_work_item_sizes_2",          "INTEGER"),
    ("max_write_image_args",           "INTEGER"),
    ("mem_base_addr_align",            "INTEGER"),
    ("min_data_type_align_size",       "INTEGER"),
    ("name",                           "TEXT", "PRIMARY KEY"),
    ("native_vector_width_char",       "INTEGER"),
    ("native_vector_width_double",     "INTEGER"),
    ("native_vector_width_float",      "INTEGER"),
    ("native_vector_width_half",       "INTEGER"),
    ("native_vector_width_int",        "INTEGER"),
    ("native_vector_width_long",       "INTEGER"),
    ("native_vector_width_short",      "INTEGER"),
    ("preferred_vector_width_char",    "INTEGER"),
    ("preferred_vector_width_double",  "INTEGER"),
    ("preferred_vector_width_float",   "INTEGER"),
    ("preferred_vector_width_half",    "INTEGER"),
    ("preferred_vector_width_int",     "INTEGER"),
    ("preferred_vector_width_long",    "INTEGER"),
    ("preferred_vector_width_short",   "INTEGER"),
    ("queue_properties",               "INTEGER"),
    ("single_fp_config",               "INTEGER"),
    ("type",                           "INTEGER"),
    ("vendor",                         "TEXT"),
    ("vendor_id",                      "TEXT"),
    ("version",                        "TEXT")
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


def vectorise_ratios(checksum, source, ratios):
    """
    Vectorise a dictionary of stencil kernel features.
    """
    vector = [checksum, source]
    for feature in KERNEL_TABLE_SCHEMA[2:]:
        # FIXME: underscores??
        if feature[0] in ratios:
            vector.append(ratios[feature[0]])
        else:
            vector.append(0)
    return vector


def vectorise_devinfo(info):
    """
    Vectorise a dictionary of OpenCL device info.
    """
    vector = [system.HOSTNAME]
    for feature in DEVICE_TABLE_SCHEMA[1:]:
        vector.append(info[feature[0]])

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


def get_source_features(checksum, source, path=""):
    user_source = get_user_source(source)
    bitcode = llvm.bitcode(user_source, path=path)
    instcounts = llvm.instcounts(bitcode, path=path)
    ratios = llvm.instcounts2ratios(instcounts)

    return vectorise_ratios(checksum, source, ratios)


def get_local_device_features():
    return [vectorise_devinfo(info) for info in opencl.get_devinfos()]


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

        # Setup persistent database.
        self.db = db.Database("/tmp/omnitune.skelcl.db")
        self.db.create_table("kernels", KERNEL_TABLE_SCHEMA)
        self.db.create_table("devices", DEVICE_TABLE_SCHEMA)
        self.db.create_table("runtime", RUNTIMES_TABLE_SCHEMA)

        # Add local device features to database.
        for info in get_local_device_features():
            self.db.insert_unique("devices", info)

    def get_source_features(self, source, checksum):
        try:
            what = ", ".join(x[0] for x in KERNEL_TABLE_SCHEMA[2:])
            where = "checksum = '{0}'".format(checksum)
            sourcefeatures = list(self.db.select1("kernels", what, where))
        except TypeError:
            sourcefeatures = get_source_features(checksum, source,
                                                 path=self.LLVM_PATH)
            self.db.insert_unique("kernels", sourcefeatures)
        return sourcefeatures

    def get_device_features(self, device_name):
        try:
            what = ", ".join([x[0] for x in DEVICE_TABLE_SCHEMA[1:]])
            where = "name = '{0}'".format(device_name)
            return list(self.db.select1("devices", what, where))
        except TypeError:
            raise FeatureExtractionError(("Failed to lookup device features for "
                                          "'{0}'".format(device_name)))

    @dbus.service.method(INTERFACE_NAME, in_signature='siiiiiiiis',
                         out_signature='(nn)')
    def RequestStencilParams(self, device_name, device_count,
                             north, south, east, west, data_width,
                             data_height, source, max_wg_size):
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
            max_wg_size: The maximum kernel workgroup size.

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
        max_wg_size = int(max_wg_size)

        # Calculate checksum of source code.
        checksum = checksum_str(source)

        sourcefeatures = self.get_source_features(source, checksum)
        devicefeatures = self.get_device_features(device_name)

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
