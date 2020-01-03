from third_party.py.pyopencl import pyopencl as cl


def get_devices():
  """
  Return a list of OpenCL devices.
  """
  devices = []
  platforms = cl.get_platforms()

  for platform in platforms:
    devices += platform.get_devices()

  return devices


def get_devinfo(device):
  info = {
    "address_bits": device.address_bits,
    "double_fp_config": device.double_fp_config,
    "endian_little": device.endian_little,
    "execution_capabilities": device.execution_capabilities,
    "extensions": device.extensions,
    "global_mem_cache_size": device.global_mem_cache_size,
    "global_mem_cache_type": device.global_mem_cache_type,
    "global_mem_cacheline_size": device.global_mem_cacheline_size,
    "global_mem_size": device.global_mem_size,
    "host_unified_memory": device.host_unified_memory,
    "image2d_max_height": device.image2d_max_height,
    "image2d_max_width": device.image2d_max_width,
    "image3d_max_depth": device.image3d_max_depth,
    "image3d_max_height": device.image3d_max_height,
    "image3d_max_width": device.image3d_max_width,
    "image_support": device.image_support,
    "local_mem_size": device.local_mem_size,
    "local_mem_type": device.local_mem_type,
    "max_clock_frequency": device.max_clock_frequency,
    "max_compute_units": device.max_compute_units,
    "max_constant_args": device.max_constant_args,
    "max_constant_buffer_size": device.max_constant_buffer_size,
    "max_mem_alloc_size": device.max_mem_alloc_size,
    "max_parameter_size": device.max_parameter_size,
    "max_read_image_args": device.max_read_image_args,
    "max_samplers": device.max_samplers,
    "max_work_group_size": device.max_work_group_size,
    "max_work_item_dimensions": device.max_work_item_dimensions,
    "max_work_item_sizes_0": device.max_work_item_sizes[0],
    "max_work_item_sizes_1": device.max_work_item_sizes[1],
    "max_work_item_sizes_2": device.max_work_item_sizes[2],
    "max_write_image_args": device.max_write_image_args,
    "mem_base_addr_align": device.mem_base_addr_align,
    "min_data_type_align_size": device.min_data_type_align_size,
    "name": device.name,
    "native_vector_width_char": device.native_vector_width_char,
    "native_vector_width_double": device.native_vector_width_double,
    "native_vector_width_float": device.native_vector_width_float,
    "native_vector_width_half": device.native_vector_width_half,
    "native_vector_width_int": device.native_vector_width_int,
    "native_vector_width_long": device.native_vector_width_long,
    "native_vector_width_short": device.native_vector_width_short,
    "preferred_vector_width_char": device.preferred_vector_width_char,
    "preferred_vector_width_double": device.preferred_vector_width_double,
    "preferred_vector_width_float": device.preferred_vector_width_float,
    "preferred_vector_width_half": device.preferred_vector_width_half,
    "preferred_vector_width_int": device.preferred_vector_width_int,
    "preferred_vector_width_long": device.preferred_vector_width_long,
    "preferred_vector_width_short": device.preferred_vector_width_short,
    "queue_properties": device.queue_properties,
    "single_fp_config": device.single_fp_config,
    "type": device.type,
    "vendor": device.vendor,
    "vendor_id": device.vendor_id,
    "version": device.version,
  }

  return info


def get_devinfos():
  """
  Return a list of device info dicts for each device on system.
  """
  return [get_devinfo(device) for device in get_devices()]


def lookup_by_name(name):
  lname = name.strip().lower()
  for device in get_devinfos():
    if device["name"].strip().lower() == lname:
      return device
  raise Exception("Could not find device '{}'".format(lname))
