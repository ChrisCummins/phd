#include "gpu/cldrive/csv_log.h"

#include "phd/logging.h"

#include <iostream>

namespace gpu {
namespace cldrive {

template <typename T>
std::ostream& NullIfZero(std::ostream& stream, const T& value) {
  if (value != 0) {
    stream << value;
  }
  return stream;
}

template <typename T>
std::ostream& NullIfEmpty(std::ostream& stream, const T& value) {
  if (!value.empty()) {
    stream << value;
  }
  return stream;
}

template <typename T>
std::ostream& NullIfNegative(std::ostream& stream, const T& value) {
  if (value >= 0) {
    stream << value;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const CsvLogHeader& header) {
  stream << "instance,device,build_opts,kernel,work_item_local_mem_size,"
         << "work_item_private_mem_size,global_size,local_size,outcome,"
         << "runtime_ns,transferred_bytes\n";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const CsvLog& log) {
  stream << log.instance_id << "," << log.device << "," << log.build_opts
         << ",";
  NullIfEmpty(stream, log.kernel) << ",";
  NullIfNegative(stream, log.work_item_local_mem_size) << ",";
  NullIfNegative(stream, log.work_item_private_mem_size) << ",";
  NullIfZero(stream, log.global_size) << ",";
  NullIfZero(stream, log.local_size) << "," << log.outcome << ",";
  NullIfZero(stream, log.runtime_ms) << ",";
  NullIfZero(stream, log.transferred_bytes) << "," << std::endl;
  return stream;
}

/*static*/ CsvLog CsvLog::FromProtos(
    int instance_id, const CldriveInstance* const instance,
    const CldriveKernelInstance* const kernel_instance,
    const CldriveKernelRun* const run,
    const gpu::libcecl::OpenClKernelInvocation* const log) {
  CsvLog csv;
  csv.instance_id = instance_id;

  CHECK(instance);
  csv.device = instance->device().name();
  csv.build_opts = instance->build_opts();
  csv.work_item_local_mem_size = -1;
  csv.work_item_private_mem_size = -1;

  csv.outcome = CldriveInstance::InstanceOutcome_Name(instance->outcome());
  if (kernel_instance) {
    csv.kernel = kernel_instance->name();
    csv.work_item_local_mem_size =
        kernel_instance->work_item_local_mem_size_in_bytes();
    csv.work_item_private_mem_size =
        kernel_instance->work_item_private_mem_size_in_bytes();

    csv.outcome = CldriveKernelInstance::KernelInstanceOutcome_Name(
        kernel_instance->outcome());
    if (run) {
      csv.outcome = CldriveKernelRun::KernelRunOutcome_Name(run->outcome());
      if (log) {
        csv.outcome = "PASS";
        csv.global_size = log->global_size();
        csv.local_size = log->local_size();
        csv.runtime_ms = log->runtime_ms();
        csv.transferred_bytes = log->transferred_bytes();
      }
    }
  }

  return csv;
}

}  // namespace cldrive
}  // namespace gpu
