import itertools
import random
import re
import time
import thread

import dbus
import dbus.service
import dbus.mainloop.glib
import gobject

import labm8 as lab
from labm8 import cache
from labm8 import crypto
from labm8 import fs
from labm8 import io
from labm8 import math as labmath
from labm8 import system

import omnitune
from omnitune import llvm
from omnitune import util

if system.HOSTNAME != "tim":
    from omnitune import opencl
else:
    from omnitune import opencl_tim as opencl

# Local imports.
import training
from . import checksum_str
from . import FeatureExtractionError
from . import hash_dataset
from . import hash_device
from . import hash_kernel
from . import hash_scenario
from db import Database
from migrate import migrate


SESSION_NAME   = "org.omnitune"
INTERFACE_NAME = "org.omnitune.skelcl"
OBJECT_PATH    = "/"


class Proxy(omnitune.Proxy):

    LLVM_PATH = fs.path("~/src/msc-thesis/skelcl/libraries/llvm/build/bin/")

    def __init__(self, *args, **kwargs):
        """
        Construct a SkelCL proxy.
        """
        # Fail if we can't find the path
        if not fs.isdir(self.LLVM_PATH):
            io.fatal("Could not find llvm path '{0}'".format(self.LLVM_PATH))

        super(Proxy, self).__init__(*args, **kwargs)
        io.info("Registered proxy %s/SkelCLProxy ..." % SESSION_NAME)

        # Setup persistent database.
        self.db = migrate(Database())

        # Create an in-memory sample strategy cache.
        self.strategies = cache.TransientCache()

        # Make a cache of local devices.
        self.local_devices = cache.TransientCache()
        for devinfo in opencl.get_devinfos():
            dev_name = devinfo["name"]
            self.local_devices[dev_name] = devinfo

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
            raise FeatureExtractionError(("Failed to lookup device "
                                          "features for {0}'"
                                          .format(device_name)))

    @dbus.service.method(INTERFACE_NAME, in_signature='siiiiiiiisss',
                         out_signature='(nn)')
    def RequestTrainingStencilParams(self, device_name, device_count,
                                     north, south, east, west, data_width,
                                     data_height, type_in, type_out, source,
                                     max_wg_size):
        """
        Request training parameter values for a SkelCL stencil operation.

        Determines the parameter values to use for a SkelCL stencil
        operation by iterating over the space of parameter values.

        Args:

            device_name (str): The name of the execution device.
            device_count (int): The number of execution devices.
            north (int): The stencil shape north direction.
            south (int): The stencil shape south direction.
            east (int): The stencil shape east direction.
            west (int): The stencil shape west direction.
            data_width (int): The number of columns of data.
            data_height (int): The number of rows of data.
            type_in (str): The input data type.
            type_out (str): The output data type.
            max_wg_size (int): The maximum kernel workgroup size.
            source (str): The stencil kernel source code.

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
        type_in = util.parse_str(type_in)
        type_out = util.parse_str(type_out)
        source = util.parse_str(source)
        max_wg_size = int(max_wg_size)

        # Get the scenario ID.
        device = hash_device(device_name, device_count)
        kernel = hash_kernel(north, south, east, west, max_wg_size, source)
        dataset = hash_dataset(data_width, data_height, type_in, type_out)
        scenario = hash_scenario(system.HOSTNAME, device, kernel, dataset)

        # Get sampling strategy.
        try:
            strategy = self.strategies[scenario]
        except KeyError:
            strategy = training.SampleStrategy(scenario, max_wg_size, self.db)
            self.strategies[scenario] = strategy

        # Get the sampling strategy's next recommendation.
        wg = strategy.next()

        end_time = time.time()

        io.debug(("RequestTrainingStencilParams({count}x {dev}, "
                  "{tout}({tin},{n},{s},{e},{w}) : {width}x{height}, "
                  "{id}, {max}) -> ({c}, {r}) [{t:.3f}s] ({p:.1f}%)"
                  .format(dev=device_name.strip()[:8],
                          count=device_count,
                          n=north, s=south, e=east, w=west,
                          width=data_width, height=data_height,
                          tin=type_in, tout=type_out,
                          id=kernel[:8], max=max_wg_size,
                          c=wg[0], r=wg[1], t=end_time - start_time,
                          p=strategy.coverage * 100)))

        return wg

    @dbus.service.method(INTERFACE_NAME, in_signature='siiiiiiiisss',
                         out_signature='(nn)')
    def RequestStencilParams(self, device_name, device_count,
                             north, south, east, west, data_width,
                             data_height, type_in, type_out, source,
                             max_wg_size):
        """
        Request parameter values for a SkelCL stencil operation.

        Determines the parameter values to use for a SkelCL stencil
        operation, using a machine learning classifier to predict the
        optimal parameter values given a set of features determined
        from the arguments.

        Args:

            device_name (str): The name of the execution device.
            device_count (int): The number of execution devices.
            north (int): The stencil shape north direction.
            south (int): The stencil shape south direction.
            east (int): The stencil shape east direction.
            west (int): The stencil shape west direction.
            data_width (int): The number of columns of data.
            data_height (int): The number of rows of data.
            type_in (str): The input data type.
            type_out (str): The output data type.
            max_wg_size (int): The maximum kernel workgroup size.
            source (str): The stencil kernel source code.

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

        # TODO: Perform feature extraction & classification
        wg = (64, 32)

        end_time = time.time()

        io.debug(("RequestStencilParams() -> "
                  "({c}, {r}) [{t:.3f}s]"
                  .format(c=wg[0], r=wg[1], t=end_time - start_time)))

        return wg

    @dbus.service.method(INTERFACE_NAME, in_signature='siiiiiiisssiiid',
                         out_signature='')
    def AddStencilRuntime(self, device_name, device_count,
                          north, south, east, west, data_width,
                          data_height, type_in, type_out, source,
                          max_wg_size, wg_c, wg_r, runtime):
        """
        Add a new stencil runtime.

        Args:

            device_name (str): The name of the execution device.
            device_count (int): The number of execution devices.
            north (int): The stencil shape north direction.
            south (int): The stencil shape south direction.
            east (int): The stencil shape east direction.
            west (int): The stencil shape west direction.
            data_width (int): The number of columns of data.
            data_height (int): The number of rows of data.
            type_in (str): The input data type.
            type_out (str): The output data type.
            source (str): The stencil kernel source code.
            max_wg_size (int): The maximum kernel workgroup size.
            wg_c (int): The workgroup size used (columns).
            wg_r (int): The workgroup size used (rows).
            runtime (double): The measured runtime in milliseconds.

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
        type_in = util.parse_str(type_in)
        type_out = util.parse_str(type_out)
        source = util.parse_str(source)
        max_wg_size = int(max_wg_size)
        wg_c = int(wg_c)
        wg_r = int(wg_r)
        runtime = float(runtime)

        # Add entry into devices table.
        devinfo = self.local_devices[device_name]
        device_id = self.db.add_device(devinfo, device_count)

        # Add entry into kernels table.
        kernel_id = self.db.add_kernel(north, south, east, west,
                                       max_wg_size, source)

        # Add entry into datasets table.
        dataset_id = self.db.add_dataset(data_width, data_height,
                                         type_in, type_out)

        # Add entry into scenarios table.
        scenario_id = self.db.add_scenario(system.HOSTNAME, device_id,
                                           kernel_id, dataset_id)

        # Add entry into params table.
        params_id = self.db.add_params(wg_c, wg_r)

        # Add entry into runtimes table.
        self.db.add_runtime(scenario_id, params_id, runtime)

        self.db.commit()

        io.debug(("AddStencilRuntime({scenario}, {params}, {runtime})"
                  .format(scenario=scenario_id[:8], params=params_id,
                          runtime=runtime)))


def main():
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    bus = dbus.SystemBus()
    name = dbus.service.BusName(SESSION_NAME, bus)
    io.info("Launched session %s ..." % SESSION_NAME)

    # Launch proxy.
    Proxy(bus, OBJECT_PATH)

    mainloop = gobject.MainLoop()
    try:
        mainloop.run()
    except KeyboardInterrupt:
        labm8.exit()
