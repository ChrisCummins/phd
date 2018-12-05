#include "datasets/me_db/health_kit/health_kit_lib.h"

#include "phd/pbutil.h"

PBUTIL_INPLACE_PROCESS_MAIN(me::ProcessHealthKitXmlExportOrDie,
                            me::SeriesCollection);
