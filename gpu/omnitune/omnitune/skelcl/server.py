import time

import dbus.mainloop.glib
import dbus.service
import gobject
import omnitune
import training
from db import Database
from migrate import migrate
from omnitune import util
from phd import labm8 as lab

from labm8.py import cache
from labm8.py import fs
from labm8.py import io

# Local imports.

SESSION_NAME = "org.omnitune"
INTERFACE_NAME = "org.omnitune.skelcl"
OBJECT_PATH = "/"


class Server(omnitune.Server):
  LLVM_PATH = fs.path("~/src/msc-thesis/skelcl/libraries/llvm/build/bin/")

  def __init__(self, *args, **kwargs):
    """
    Construct a SkelCL server.
    """
    # Fail if we can't find the path
    if not fs.isdir(self.LLVM_PATH):
      io.fatal("Could not find llvm path '{0}'".format(self.LLVM_PATH))

    super(Server, self).__init__(*args, **kwargs)
    io.info("Registered server %s/SkelCLServer ..." % SESSION_NAME)

    # Setup persistent database.
    self.db = migrate(Database())
    self.db.status_report()

    # Create an in-memory sample strategy cache.
    self.strategies = cache.TransientCache()

  @dbus.service.method(
    INTERFACE_NAME, in_signature="siiiiiiiisss", out_signature="(nn)"
  )
  def RequestTrainingStencilParams(
    self,
    device_name,
    device_count,
    north,
    south,
    east,
    west,
    data_width,
    data_height,
    type_in,
    type_out,
    source,
    max_wg_size,
  ):
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

    # Get the next scenario ID to train on.
    wg = training.random_wg_value(max_wg_size)

    return wg

  @dbus.service.method(
    INTERFACE_NAME, in_signature="siiiiiiiisss", out_signature="(nn)"
  )
  def RequestStencilParams(
    self,
    device_name,
    device_count,
    north,
    south,
    east,
    west,
    data_width,
    data_height,
    type_in,
    type_out,
    source,
    max_wg_size,
  ):
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

    io.debug(
      (
        "RequestStencilParams() -> "
        "({c}, {r}) [{t:.3f}s]".format(
          c=wg[0], r=wg[1], t=end_time - start_time
        )
      )
    )

    return wg

  @dbus.service.method(
    INTERFACE_NAME, in_signature="siiiiiiisssiiid", out_signature=""
  )
  def AddStencilRuntime(
    self,
    device_name,
    device_count,
    north,
    south,
    east,
    west,
    data_width,
    data_height,
    type_in,
    type_out,
    source,
    max_wg_size,
    wg_c,
    wg_r,
    runtime,
  ):
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

    # Lookup IDs
    device = self.db.device_id(device_name, device_count)
    kernel = self.db.kernel_id(north, south, east, west, max_wg_size, source)
    dataset = self.db.datasets_id(data_width, data_height, type_in, type_out)
    scenario = self.db.scenario_id(device, kernel, dataset)
    params = self.db.params_id(wg_c, wg_r)

    # Add entry into runtimes table.
    self.db.add_runtime(scenario, params, runtime)
    self.db.commit()

    io.debug(
      (
        "AddStencilRuntime({scenario}, {params}, {runtime})".format(
          scenario=scenario[:8], params=params, runtime=runtime
        )
      )
    )

  @dbus.service.method(
    INTERFACE_NAME, in_signature="siiiiiiisssiii", out_signature=""
  )
  def RefuseStencilParams(
    self,
    device_name,
    device_count,
    north,
    south,
    east,
    west,
    data_width,
    data_height,
    type_in,
    type_out,
    source,
    max_wg_size,
    wg_c,
    wg_r,
  ):
    """
    Mark a set of parameters as bad.

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

    # Lookup IDs
    device = self.db.device_id(device_name, device_count)
    kernel = self.db.kernel_id(north, south, east, west, max_wg_size, source)
    dataset = self.db.datasets_id(data_width, data_height, type_in, type_out)
    scenario = self.db.scenario_id(device, kernel, dataset)
    params = self.db.params_id(wg_c, wg_r)

    # Add entry into runtimes table.
    self.db.refuse_params(scenario, params)
    self.db.commit()

    io.debug(
      (
        "RefuseStencilParams({scenario}, {params})".format(
          scenario=scenario[:8], params=params, runtime=runtime
        )
      )
    )


def main():
  dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

  bus = dbus.SystemBus()
  name = dbus.service.BusName(SESSION_NAME, bus)
  io.info("Launched session %s ..." % SESSION_NAME)

  # Launch server.
  Server(bus, OBJECT_PATH)

  mainloop = gobject.MainLoop()
  try:
    mainloop.run()
  except KeyboardInterrupt:
    lab.exit()
