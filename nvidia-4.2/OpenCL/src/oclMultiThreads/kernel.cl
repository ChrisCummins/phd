__kernel void
simpleIncrement( __global float* C)
{
	C[get_global_id(0)] += 1.0f;
}