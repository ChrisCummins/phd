#include "phd/app.h"

namespace phd {

void InitApp(int* argc, char*** argv, const char* usage_string) {
  std::string usage(usage_string);
  if (!usage.empty()) {
    gflags::SetUsageMessage(usage);
  }

  gflags::ParseCommandLineFlags(argc, argv, /*remove_flags=*/true);
}

}  // namespace phd