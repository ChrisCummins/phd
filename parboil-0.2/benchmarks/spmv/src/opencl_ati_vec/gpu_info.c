/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <endian.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <inttypes.h>

#include "gpu_info.h"

/*
 * Workgroup is multiple of 64 threads
 * Max threads 265
 */
void compute_active_thread(size_t *thread,
			   size_t *grid,
			   int task,
			   int pad)
{
	int max_thread=496*64;
	int max_block=256;
	int _grid;
	int _thread;
	
	if(task*pad>max_thread)
	{
		_thread=max_block;
		_grid = ((task*pad+_thread-1)/_thread)*_thread;
	}
	else
	{
		_thread=pad;
		_grid=task*pad;
	}

	thread[0]=_thread;
	grid[0]=_grid;
}
