#include "labm8/cpp/logging.h"
#include "labm8/cpp/pbutil.h"
#include "learn/python/cpp_interop.pb.h"

// The "work" function which processes an input message in place.
void ProcessProtobufInPlace(AddXandY* proto) {
  int x = proto->x();
  int y = proto->y();
  LOG(DEBUG) << "Adding " << x << " and " << y;
  proto->set_result(x + y);
}

PBUTIL_INPLACE_PROCESS_MAIN(ProcessProtobufInPlace, AddXandY);
