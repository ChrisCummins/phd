ATI Optimized Sparse Matrix Vector Mult
Ian Wetherbee


To try:
Vectorize access to:
	sparse matrix values
	sparse matrix locations
Normal loading of vector values into a float4
Loop unroll in groups of 4 to fill all 5 lanes of VLIW
	Vec4 operations are already "unrolled" as they fill 4 lanes
With 64 threads having access to the same local memory, would it make
	sense to try and load the full vector into local? Probably not,
	but it depends how "sparse" the matrix really is
Load full vector into 2D image memory
Load jds_ptr_int into local memory

To implement:
	Group row items into vec4s, transpose rows->cols and arrange in col-
		major order. Each thread loops "down" a row.
	Pad each transposed horizontal column to a multiple of 256 bytes
	Pad full vector to a multiple of 4.

Use ternary x = (if) ? yes : no; as it doesn't diverge
	Causes a static overhead even if no thread follows one condition
	Better than a clause switch caused by an if() statement in most cases

256 thread limit per workgroup
16/64 threads execute at a time on a CU (4 cycles per instruction)
Pick X dimensions of at least 16
Memory accesses:
	265 byte bursts, so use vectors as much as possible
	Pad sparse matrix values and locations to groups of 4
	Wavefront coallesing - 8 memory channels, 64 threads
	Broadcasts anywhere?
		Broadcasts don't work from global memory and actually
		cause a serialization of accesses to the same channel
		Instead, have only one thread read to local memory
	Take advantage of cache hits?

Global memory optimization:
	-----------------------------------------------------------------------
	| [31:x]  |      bank            |  [11:8] channel   |  [7:0] address |
        -----------------------------------------------------------------------
	channel = (address / 265) % n
	n = 8 on the 5870
	FastPath memory - used for simple loads/stores of 32*n bits
		CompletePath is triggered by things like unaligned reads/writes
		or atomic operations
	Each wavefront should access consecutive groups of 256 bytes
		=> each wavefront's threads should access memory from different
		channels (in strides of 256B)
	Each wavefront should NOT be a power of 2 apart

Constant memory optimizations:
	Broadcast when all threads access same constant address
	Varying-index accesses have the same time as a global access

Compute unit (CU) specs (5000 series):
	32kB local memory
	L1 cache - 8kB per CU
	Register file
	16 processing units - 5-way VLIW

