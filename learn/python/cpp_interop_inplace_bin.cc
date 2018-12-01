#include "learn/python/cpp_interop.pb.h"
#include "phd/macros.h"
#include "phd/pbutil.h"

// The "work" function which processes an input message in place.
void ProcessProtobufInPlace(AddXandY* proto) {
  int x = proto->x();
  int y = proto->y();
  DEBUG("Adding %d and %d", x, y);
  proto->set_result(x + y);
}

PBUTIL_INPLACE_PROCESS_MAIN(ProcessProtobufInPlace, AddXandY);
