#include "gpu/cldrive/libcldrive.h"
#include "gpu/cldrive/proto/cldrive.pb.h"

#include "phd/pbutil.h"

PBUTIL_INPLACE_PROCESS_MAIN(gpu::cldrive::ProcessCldriveInstanceOrDie,
                            gpu::cldrive::CldriveInstance);
