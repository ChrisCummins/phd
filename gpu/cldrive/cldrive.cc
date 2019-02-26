#include "gpu/cldrive/libcldrive.h"

#include "gpu/cldrive/proto/cldrive.pb.h"

#include "gpu/clinfo/libclinfo.h"

#include "phd/logging.h"

#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "gflags/gflags.h"

#include <sstream>

DEFINE_string(src, "", "Path to a file containing OpenCL kernels.");
static bool ValidateSrc(const char* flagname, const string& value) {
  if (value.empty()) {
    LOG(FATAL) << "Flag --" << flagname << " must be set";
  }

  boost::filesystem::path path(value);
  if (!boost::filesystem::is_regular_file(path)) {
    LOG(FATAL) << "File not found: " << value;
  }

  return true;
}
DEFINE_validator(src, &ValidateSrc);

DEFINE_string(env, "",
              "Specify the OpenCL device to use. Run `bazel run //gpu/clinfo` "
              "to see a list of available devices. Defaults to using the "
              "builtin CPU simulator.");
static bool ValidateEnv(const char* flagname, const string& value) {
  if (value.empty()) {
    LOG(FATAL) << "Flag --" << flagname << " must be set";
  }
  try {
    phd::gpu::clinfo::GetOpenClDevice(value);
  } catch (std::invalid_argument e) {
    LOG(ERROR) << "Available OpenCL environments:";
    auto devices = phd::gpu::clinfo::GetOpenClDevices();
    for (int i = 0; i < devices.device_size(); ++i) {
      LOG(ERROR) << "    " << devices.device(i).name();
    }
    LOG(FATAL) << "OpenCL environment not found";
  }
  return true;
}
DEFINE_validator(env, &ValidateEnv);

DEFINE_int32(gsize, 64, "The global size to use.");
DEFINE_int32(lsize, 32, "The local (workgroup) size.");
DEFINE_bool(cl_opt, true, "Whether OpenCL optimizations are enabled.");
DEFINE_int32(num_runs, 10, "The number of runs per kernel.");
DEFINE_bool(binary_output, false, "Whether to print binary protobuf to stdout.");

namespace gpu {
namespace cldrive {


}  // namespace cldrive
}  // namespace gpu

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Read OpenCL source.
  const boost::filesystem::path src_path(FLAGS_src);
  CHECK(boost::filesystem::is_regular_file(src_path));
  boost::filesystem::ifstream istream(src_path);
  CHECK(istream.is_open());

  std::stringstream buffer;
  buffer << istream.rdbuf();

  auto device = phd::gpu::clinfo::GetOpenClDeviceOrDie(FLAGS_env);

  // Setup instance proto.
  gpu::cldrive::CldriveInstance instance;
  instance.set_opencl_src(buffer.str());
  auto dp = instance.add_dynamic_params();
  dp->set_global_size_x(FLAGS_gsize);
  dp->set_local_size_x(FLAGS_lsize);
  instance.set_min_runs_per_kernel(FLAGS_num_runs);

  gpu::cldrive::Cldrive(&instance, device).RunOrDie();

  if (FLAGS_binary_output) {
    instance.SerializeToOstream(&std::cout);
  } else {
    std::cout << "# File: //gpu/cldrive/proto/cldrive.proto\n"
              << "# Proto: gpu.cldrive.CldriveInstance\n"
              << instance.DebugString();
  }
}
