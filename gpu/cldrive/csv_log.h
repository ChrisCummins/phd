#pragma once

#include "gpu/cldrive/proto/cldrive.pb.h"
#include "phd/port.h"
#include "phd/string.h"

#include <vector>

namespace gpu {
namespace cldrive {

class CsvLogHeader {
  friend std::ostream& operator<<(std::ostream& stream,
                                  const CsvLogHeader& log);
};

class CsvLog {
 public:
  CsvLog() = default;

  // From CldriveInstance.opencl_src field.
  int instance_id;

  // From CldriveInstance.device.name field.
  string device;

  // From CldriveInstance.build_opts field.
  string build_opts;

  // The OpenCL kernel name, from CldriveKernelInstance.name
  // field. If CldriveInstance.outcome != PASS, this will be empty.
  string kernel;

  // From CldriveKernelInstance.work_item_local_mem_size_in_bytes field. If
  // CldriveInstance.outcome != PASS, this will be empty.
  int work_item_local_mem_size;

  // From CldriveKernelInstance.work_item_private_mem_size_in_bytes field. If
  // CldriveInstance.outcome != PASS, this will be empty.
  int work_item_private_mem_size;

  // From CldriveInstance.dynamic_params.global_size_x field. If
  // CldriveInstance.outcome != PASS, this will be empty.
  int global_size;

  // From CldriveInstance.dynamic_params.local_size_x field. If
  // CldriveInstance.outcome != PASS, this will be empty.
  int local_size;

  // A stringified enum value. Either CldriveInstance.outcome if
  // CldriveInstance.outcome != PASS, else CldriveKernelInstance.outcome if
  // CldriveKernelInstance.outcome != PASS, else CldriveKernelRun.outcome.
  string outcome;

  // From CldriveKernelRun.log.runtime_ms. If outcome != PASS, this will be
  // empty.
  double runtime_ms;

  // From CldriveKernelRun.log.transferred_bytes. If outcome != PASS, this will
  // be empty.
  phd::int64 transferred_bytes;

  static CsvLog FromProtos(int instance_id, CldriveInstance* instance,
                           CldriveKernelInstance* kernel_instance,
                           CldriveKernelRun* run,
                           gpu::libcecl::OpenClKernelInvocation* log);

  friend std::ostream& operator<<(std::ostream& stream, const CsvLog& log);
};

std::ostream& operator<<(std::ostream& stream, const CsvLogHeader& log);
std::ostream& operator<<(std::ostream& stream, const CsvLog& log);

}  // namespace cldrive
}  // namespace gpu
