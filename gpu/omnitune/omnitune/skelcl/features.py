from __future__ import division
from __future__ import print_function

from omnitune import llvm

from labm8 import system

if system.HOSTNAME == "tim" or system.HOSTNAME == "zoo":
  from omnitune import opencl_tim as opencl
else:
  from omnitune import opencl


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


def kernel(north, south, east, west, max_wg_size, source):
  """
  Perform feature extraction on a kernel.
  """
  bitcode = llvm.bitcode(source)
  instcounts = llvm.instcounts(bitcode)
  ratios = llvm.instcounts2ratios(instcounts)
  return (
      north,  # north
      south,  # south
      east,  # east
      west,  # west
      max_wg_size,  # max_wg_size
      ratios.get("instruction_count", 0),  # instruction_count
      ratios.get("ratio AShr insts", 0),  # ratio_AShr_insts
      ratios.get("ratio Add insts", 0),  # ratio_Add_insts
      ratios.get("ratio Alloca insts", 0),  # ratio_Alloca_insts
      ratios.get("ratio And insts", 0),  # ratio_And_insts
      ratios.get("ratio Br insts", 0),  # ratio_Br_insts
      ratios.get("ratio Call insts", 0),  # ratio_Call_insts
      ratios.get("ratio FAdd insts", 0),  # ratio_FAdd_insts
      ratios.get("ratio FCmp insts", 0),  # ratio_FCmp_insts
      ratios.get("ratio FDiv insts", 0),  # ratio_FDiv_insts
      ratios.get("ratio FMul insts", 0),  # ratio_FMul_insts
      ratios.get("ratio FPExt insts", 0),  # ratio_FPExt_insts
      ratios.get("ratio FPToSI insts", 0),  # ratio_FPToSI_insts
      ratios.get("ratio FSub insts", 0),  # ratio_FSub_insts
      ratios.get("ratio GetElementPtr insts", 0),  # ratio_GetElementPtr_insts
      ratios.get("ratio ICmp insts", 0),  # ratio_ICmp_insts
      ratios.get("ratio InsertValue insts", 0),  # ratio_InsertValue_insts
      ratios.get("ratio Load insts", 0),  # ratio_Load_insts
      ratios.get("ratio Mul insts", 0),  # ratio_Mul_insts
      ratios.get("ratio Or insts", 0),  # ratio_Or_insts
      ratios.get("ratio PHI insts", 0),  # ratio_PHI_insts
      ratios.get("ratio Ret insts", 0),  # ratio_Ret_insts
      ratios.get("ratio SDiv insts", 0),  # ratio_SDiv_insts
      ratios.get("ratio SExt insts", 0),  # ratio_SExt_insts
      ratios.get("ratio SIToFP insts", 0),  # ratio_SIToFP_insts
      ratios.get("ratio SRem insts", 0),  # ratio_SRem_insts
      ratios.get("ratio Select insts", 0),  # ratio_Select_insts
      ratios.get("ratio Shl insts", 0),  # ratio_Shl_insts
      ratios.get("ratio Store insts", 0),  # ratio_Store_insts
      ratios.get("ratio Sub insts", 0),  # ratio_Sub_insts
      ratios.get("ratio Trunc insts", 0),  # ratio_Trunc_insts
      ratios.get("ratio UDiv insts", 0),  # ratio_UDiv_insts
      ratios.get("ratio Xor insts", 0),  # ratio_Xor_insts
      ratios.get("ratio ZExt insts", 0),  # ratio_ZExt_insts
      ratios.get("ratio basic blocks", 0),  # ratio_basic_blocks
      ratios.get("ratio memory instructions", 0),  # ratio_memory_instructions
      ratios.get("ratio non-external functions", 0)
      # ratio_non_external_functions
  )


def device(name, count):
  """
  Perform feature extraction on a device.
  """

  def _get_devinfo(name):
    for devinfo in opencl.get_devinfos():
      if devinfo["name"] == name:
        return devinfo
    raise FeatureExtractionError("Device '" + name + "' not found in "
                                 "local device info")

  devinfo = _get_devinfo(name)
  return (
      name, count, devinfo["address_bits"], devinfo["double_fp_config"],
      devinfo["endian_little"], devinfo["execution_capabilities"],
      devinfo["extensions"], devinfo["global_mem_cache_size"],
      devinfo["global_mem_cache_type"], devinfo["global_mem_cacheline_size"],
      devinfo["global_mem_size"], devinfo["host_unified_memory"],
      devinfo["image2d_max_height"], devinfo["image2d_max_width"],
      devinfo["image3d_max_depth"], devinfo["image3d_max_height"],
      devinfo["image3d_max_width"], devinfo["image_support"],
      devinfo["local_mem_size"], devinfo["local_mem_type"],
      devinfo["max_clock_frequency"], devinfo["max_compute_units"],
      devinfo["max_constant_args"], devinfo["max_constant_buffer_size"],
      devinfo["max_mem_alloc_size"], devinfo["max_parameter_size"],
      devinfo["max_read_image_args"], devinfo["max_samplers"],
      devinfo["max_work_group_size"], devinfo["max_work_item_dimensions"],
      devinfo["max_work_item_sizes_0"], devinfo["max_work_item_sizes_1"],
      devinfo["max_work_item_sizes_2"], devinfo["max_write_image_args"],
      devinfo["mem_base_addr_align"], devinfo["min_data_type_align_size"],
      devinfo["native_vector_width_char"],
      devinfo["native_vector_width_double"],
      devinfo["native_vector_width_float"], devinfo["native_vector_width_half"],
      devinfo["native_vector_width_int"], devinfo["native_vector_width_long"],
      devinfo["native_vector_width_short"],
      devinfo["preferred_vector_width_char"],
      devinfo["preferred_vector_width_double"],
      devinfo["preferred_vector_width_float"],
      devinfo["preferred_vector_width_half"],
      devinfo["preferred_vector_width_int"],
      devinfo["preferred_vector_width_long"],
      devinfo["preferred_vector_width_short"], devinfo["queue_properties"],
      devinfo["single_fp_config"], devinfo["type"], devinfo["vendor"],
      devinfo["vendor_id"], devinfo["version"])


def dataset(*features):
  return features
