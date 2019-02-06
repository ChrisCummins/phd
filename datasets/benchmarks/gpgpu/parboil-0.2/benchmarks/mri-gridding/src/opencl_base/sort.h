/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <CL/cl.h>

void sort (int numElems, unsigned int max_value, cl_mem* &dkeys, cl_mem* &dvalues, cl_mem* &dkeys_o, cl_mem* &dvalues_o, cl_context *clContext, cl_command_queue clCommandQueue, const cl_device_id clDevice, size_t *workItemSizes);
