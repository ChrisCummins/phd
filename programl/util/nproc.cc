#include "programl/util/nproc.h"

#include <string>
#include "subprocess/subprocess.hpp"

namespace programl {
namespace util {

size_t GetNumberOfProcessors() {
  auto out = subprocess::check_output("nproc");
  return std::stoi(std::string(out.buf.begin(), out.buf.end()));
}

}  // namespace util
}  // namespace programl
