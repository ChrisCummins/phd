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
  stream << ",";
  return stream;
}

template <typename T>
std::ostream& NullIfEmpty(std::ostream& stream, const T& value) {
  if (!value.empty()) {
    stream << value;
  }
  stream << ",";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const CsvLogHeader& header) {
  stream << "instance,device,build_opts,kernel,work_item_local_mem_size,"
         << "work_item_private_mem_size,global_size,local_size,outcome,"
         << "runtime_ns,transferred_bytes";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const CsvLog& log) {
  stream << log.instance_id << "," << log.device << "," << log.build_opts
         << ",";
  NullIfEmpty(stream, log.kernel) << ",";
  NullIfZero(stream, log.work_item_local_mem_size);
  NullIfZero(stream, log.work_item_private_mem_size);
  NullIfZero(stream, log.global_size);
  NullIfZero(stream, log.local_size) << log.outcome << ",";
  NullIfZero(stream, log.runtime_ns);
  NullIfZero(stream, log.transferred_bytes) << std::endl;
  return stream;
}

static CsvLog FromProtos(int instance_id, CldriveInstance* instance,
                         CldriveKernelInstance* kernel_instance,
                         CldriveKernelRun* run,
                         gpu::libcecl::OpenClKernelInvocation* log) {
  CsvLog csv;
  csv.instance_id = instance_id;

  CHECK(instance);
  csv.device = instance->device().name();
  csv.build_opts = instance->build_opts();

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
