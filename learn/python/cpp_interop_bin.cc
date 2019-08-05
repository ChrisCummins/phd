#include "labm8/cpp/logging.h"
#include "labm8/cpp/pbutil.h"
#include "learn/python/cpp_interop.pb.h"

void ProcessProtobuf(const AddXandY& input_proto, AddXandY* output_proto) {
  int x = input_proto.x();
  int y = input_proto.y();
  LOG(DEBUG) << "Adding " << x << " and " << y << " and storing the result in "
             << "a new message";
  output_proto->set_result(x + y);
}

PBUTIL_PROCESS_MAIN(ProcessProtobuf, AddXandY, AddXandY);
