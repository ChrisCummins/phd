#include "learn/python/cpp_interop.pb.h"
#include "phd/logging.h"
#include "phd/pbutil.h"

// The "work" function which processes an input message in place.
void ProcessProtobufInPlace(AddXandY* proto) {
  int x = proto->x();
  int y = proto->y();
  LOG(DEBUG) << "Adding " << x << " and " << y;
  proto->set_result(x + y);
}

PBUTIL_INPLACE_PROCESS_MAIN(ProcessProtobufInPlace, AddXandY);
