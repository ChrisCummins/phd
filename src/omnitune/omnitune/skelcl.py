import itertools
import random
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


KERNELS_TABLE = (
    ("checksum",                     "TEXT", "PRIMARY KEY"),
    ("source",                       "TEXT")
)

KERNEL_FEATURES_TABLE = (
    ("checksum",                     "TEXT", "PRIMARY KEY"),
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


DEVICES_TABLE = (
    ("name",                           "TEXT", "PRIMARY KEY"),
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

RUNTIMES_TABLE = (
    ("host",                           "TEXT"),
    ("dev_name",                       "TEXT"),
    ("dev_count",                      "INTEGER"),
    ("kern_checksum",                  "TEXT"),
    ("north",                          "INTEGER"),
    ("south",                          "INTEGER"),
    ("east",                           "INTEGER"),
    ("west",                           "INTEGER"),
    ("data_width",                     "INTEGER"),
    ("data_height",                    "INTEGER"),
    ("max_wg_size",                    "INTEGER"),
    ("wg_c",                           "INTEGER"),
    ("wg_r",                           "INTEGER"),
    ("runtime",                        "REAL")
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
    for column in KERNEL_FEATURES_TABLE[1:]:
        column_name = column[0]
        # FIXME: underscores??
        if column_name in ratios:
            vector.append(ratios[column_name])
        else:
            vector.append(0)
    return vector


def vectorise_devinfo(info):
    """
    Vectorise a dictionary of OpenCL device info.
    """
    vector = [
        info[DEVICES_TABLE[0][0]],
        system.HOSTNAME
    ]
    for column in DEVICES_TABLE[2:]:
        column_name = column[0]
        vector.append(info[column_name])

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


class SkelCLDatabase(db.Database):
    """
    Persistent database store for SkelCL OmniTune data.

    Tables:
        kernels  Table of kernel features (extracted from LLVM IR).
        devices  Table of device features (extracted from OpenCL API).
        runtimes Table of offline training data.
    """

    def __init__(self, path=None):
        """
        Create a new connection to database.

        Arguments:
           path (optional) If set, load database from path. If not, use
               standard system-wide default path.
        """
        if path is None:
            path = fs.path(omnitune.LOCAL_DIR, "skelcl.db")

        tables = {
            "kernels":         KERNELS_TABLE,
            "kernel_features": KERNEL_FEATURES_TABLE,
            "devices":         DEVICES_TABLE,
            "runtimes":        RUNTIMES_TABLE
        }

        super(SkelCLDatabase, self).__init__(path, tables)


    def get_device_info(self, device_name):
        """
        Lookup info for a device.
        """
        what = ", ".join(["name"] + [x[0] for x in DEVICES_TABLE[2:]])
        where = "name = '{0}'".format(device_name)
        return list(self.select1("devices", what, where))

    def add_device_info(self, *args):
        """
        Add a new row of device info.
        """
        self.insert_unique("devices", args)


    def get_kernel_source(self, checksum):
        """
        Lookup kernel source.
        """
        what = "source"
        where = "checkum = '{0}'".format(checksum)
        return list(self.select1("kernels", what, where))

    def add_kernel_source(self, checksum, source):
        """
        Add a new kernel source code.
        """
        self.insert_unique("kernels", (checksum, source))

    def get_kernel_info(self, checksum):
        """
        Lookup info for a kernel.
        """
        what = ", ".join(x[0] for x in KERNEL_FEATURES_TABLE[1:])
        where = "checksum = '{0}'".format(checksum)
        return list(self.select1("kernels", what, where))

    def add_kernel_info(self, source_info):
        """
        Add a new row of kernel info.
        """
        self.insert_unique("kernels", source_info)

    def add_runtime(self, *args):
        """
        Add a new measured experimental runtime.
        """
        self.insert("runtimes", args)


class StencilSamplingStrategy(object):

    param_values = [4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]
    unconstrained_space = list(itertools.product(param_values,
                                                 param_values))

    def __init__(self, device_name, device_count,
                 checksum, north, south, east, west,
                 data_width, data_height,
                 max_wg_size, db):
        self.db = db

        # Generate the sample space.
        self.sample_space = [x for x in self.unconstrained_space
                             if x[0] * x[1] < max_wg_size]

        for point in self.sample_space:
            # Get the number of samples at each point in the sample
            # space.
            params = [
                system.HOSTNAME,
                device_name,
                device_count,
                checksum,
                north, south, east, west,
                data_width, data_height,
                max_wg_size,
                point[0], point[1]
            ]

            where = []
            for i in range(len(params)):
                where.append(RUNTIMES_TABLE[i][0] + "=" +
                             str(db.escape_value("runtimes", i, params[i])))

        self.where = "(" + " AND ".join(where) + ")"

        # Generate list of samples.
        self._wgs = []
        self._update_wgs()

    def _update_wgs(self):
        sample_counts = []

        io.debug("Creating sample list...")

        for point in self.sample_space:
            sample_count = self.db.count("runtimes", self.where)
            sample_counts.append((point, sample_count))

        most_samples = max([x[1] for x in sample_counts]) + 5

        jobs = []
        for sample in sample_counts:
            wg = sample[0]
            count = sample[1]
            diff = most_samples - count
            for i in range(diff):
                self._wgs.append(wg)

        random.shuffle(self._wgs)

        possible = len(sample_counts) * 250
        total = sum([x[1] for x in sample_counts])
        self.coverage = float(total) / float(possible)


    def next(self):
        if not len(self._wgs):
            self._update_wgs()

        return self._wgs.pop(0)


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
        self.db = SkelCLDatabase()

        # Create an in-memory sample strategy cache.
        self.strategies = cache.TransientCache()

        # Add local device features to database.
        for info in get_local_device_features():
            self.db.add_device_info(*info)

    def get_source_features(self, source, checksum):
        try:
            return self.db.get_kernel_info(checksum)
        except TypeError:
            sourcefeatures = get_source_features(checksum, source,
                                                 path=self.LLVM_PATH)
            self.db.add_kernel_info(*sourcefeatures)
            return sourcefeatures

    def get_device_features(self, device_name):
        try:
            return self.db.get_device_info(device_name)
        except TypeError:
            raise FeatureExtractionError(("Failed to lookup device features for "
                                          "'{0}'".format(device_name)))

    @dbus.service.method(INTERFACE_NAME, in_signature='siiiiiiiis',
                         out_signature='(nn)')
    def RequestTrainingStencilParams(self, device_name, device_count,
                                     north, south, east, west, data_width,
                                     data_height, source, max_wg_size):
        """
        Request training parameter values for a SkelCL stencil operation.

        Determines the parameter values to use for a SkelCL stencil
        operation by iterating over the space of parameter values.

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

        # Record kernel source.
        self.db.add_kernel_source(checksum, source)

        # Get sampling strategy.
        strategy_id = "".join([str(x) for x in (
            device_name, device_count, checksum,
            north, south, east, west,
            data_width, data_height,
            max_wg_size, checksum)])
        strategy = self.strategies.get(strategy_id)
        if strategy is None:
            strategy = StencilSamplingStrategy(device_name, device_count,
                                               checksum, north, south, east,
                                               west, data_width, data_height,
                                               max_wg_size, self.db)
            self.strategies.set(strategy_id, strategy)

        # Get the sampling strategy's next recommendation.
        wg = strategy.next()

        end_time = time.time()

        io.debug(("RequestTrainingStencilParams({dev}, {count}, "
                  "[{n}, {s}, {e}, {w}], {width}, {height}, {id}, {max}) -> "
                  "({c}, {r}) [{t:.3f}s] ({p:.1f}%)"
                  .format(dev=device_name.strip()[:8],
                          count=device_count,
                          n=north, s=south, e=east, w=west,
                          width=data_width, height=data_height,
                          id=checksum[:8], max=max_wg_size,
                          c=wg[0], r=wg[1], t=end_time - start_time,
                          p=strategy.coverage * 100)))

        return wg

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
                  "[{n}, {s}, {e}, {w}], {width}, {height}, {id}, {max}) -> "
                  "({c}, {r}) [{t:.3f}s]"
                  .format(dev=device_name.strip()[:8],
                          count=device_count,
                          n=north, s=south, e=east, w=west,
                          width=data_width, height=data_height,
                          id=checksum[:8], max=max_wg_size,
                          c=wg[0], r=wg[1], t=end_time - start_time)))

        return wg

    @dbus.service.method(INTERFACE_NAME, in_signature='siiiiiiisiiid',
                         out_signature='')
    def AddStencilRuntime(self, device_name, device_count,
                          north, south, east, west, data_width,
                          data_height, source, wg_c, wg_r,
                          max_wg_size, runtime):
        """
        Add a new stencil runtime.

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
            wg_c: The workgroup size (columns).
            wg_r: The workgroup size (rows).
            max_wg_size: The maximum kernel workgroup size.
            runtime: The measured kernel runtime.
        """

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
        wg_c = int(wg_c)
        wg_r = int(wg_r)
        max_wg_size = int(max_wg_size)
        runtime = float(runtime)

        # Calculate checksum of source code.
        checksum = checksum_str(source)

        self.db.add_runtime(system.HOSTNAME, device_name, device_count,
                            checksum, north, south, east, west,
                            data_width, data_height, max_wg_size,
                            wg_c, wg_r, runtime)

        io.debug(("AddStencilRuntime({dev}, {count}, "
                  "[{n}, {s}, {e}, {w}], {width}, {height}, {id}, {max}, "
                  "{c}, {r}, {t})"
                  .format(dev=device_name.strip()[:8],
                          count=device_count,
                          n=north, s=south, e=east, w=west,
                          width=data_width, height=data_height,
                          id=checksum[:8], max=max_wg_size,
                          c=wg_c, r=wg_r, t=runtime)))


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
