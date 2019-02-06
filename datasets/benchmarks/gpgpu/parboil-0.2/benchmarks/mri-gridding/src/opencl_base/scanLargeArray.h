/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <CL/cl.h>

void scanLargeArray( unsigned int gridNumElements, cl_mem data_d, cl_context clContext, cl_command_queue clCommandQueue, const cl_device_id clDevice, size_t *workItemSizes);
