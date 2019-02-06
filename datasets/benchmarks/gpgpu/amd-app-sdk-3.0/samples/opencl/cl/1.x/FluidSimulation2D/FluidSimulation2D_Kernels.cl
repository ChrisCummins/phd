/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

//
// Data required
// Global array : 9 f input values (rho and u are computed from these values)
// Global array : 9 f output values
// Constant array : Boundary or Fluid (1 bit : 0 for fluid and 1 for boundary)
// Private variables : 9 f input values, rho, u[2], 
// Constant arrays : 9 directions, 9 weights, omega, 


#ifdef KHR_DP_EXTENSION
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

// Calculates equivalent distribution 
double computefEq(double rho, double weight, double2 dir, double2 u)
{
    double u2 = (u.x * u.x) + (u.y * u.y);		//x^2 + y^2
    double eu = (dir.x * u.x) + (dir.y * u.y);	//
    return rho * weight * (1.0f + (3.0f * eu) + (4.5f * eu * eu) - (1.5f * u2));
}

__kernel void lbm(__global double *if0, __global double *of0, 
                  __global double4 *if1234, __global double4 *of1234,
                  __global double4 *if5678, __global double4 *of5678,
                  __global bool *type,	// This will only work for sizes <= 512 x 512 as constant buffer is only 64KB
                  double8 dirX,  double8 dirY,	//Directions is (0, 0) for 0
                  __constant double weight[9],	//Directions : 0, 1, 2, 3, 4, 5, 6, 7, 8
                  double omega,
                  __global double2 *velocityBuffer)
{
    uint2 id = (uint2)(get_global_id(0), get_global_id(1));
    uint width = get_global_size(0);
    uint pos = id.x + width * id.y;

    // Read input distributions
    double f0 = if0[pos];
    double4 f1234 = if1234[pos];
    double4 f5678 = if5678[pos];


    double rho;	//Density
    double2 u;	//Velocity

    // Collide
    //boundary
    if(type[pos])
    {
        // Swap directions by swizzling
        f1234.xyzw = f1234.zwxy;
        f5678.xyzw = f5678.zwxy;

        rho = 0;
        u = (double2)(0, 0);
    }
    //fluid
    else
    {
        // Compute rho and u
        // Rho is computed by doing a reduction on f
        double4 temp = f1234 + f5678;
        temp.lo += temp.hi;
        rho = temp.x + temp.y;
        rho += f0;

        // Compute velocity
        u.x = (dot(f1234, dirX.lo) + dot(f5678, dirX.hi)) / rho;
        u.y = (dot(f1234, dirY.lo) + dot(f5678, dirY.hi)) / rho;

        double4 fEq1234;	// Stores feq 
        double4 fEq5678;
        double fEq0;

        // Compute fEq
        fEq0 = computefEq(rho, weight[0], (double2)(0, 0), u);
        fEq1234.x = computefEq(rho, weight[1], (double2)(dirX.s0, dirY.s0), u);
        fEq1234.y = computefEq(rho, weight[2], (double2)(dirX.s1, dirY.s1), u);
        fEq1234.z = computefEq(rho, weight[3], (double2)(dirX.s2, dirY.s2), u);
        fEq1234.w = computefEq(rho, weight[4], (double2)(dirX.s3, dirY.s3), u);
        fEq5678.x = computefEq(rho, weight[5], (double2)(dirX.s4, dirY.s4), u);
        fEq5678.y = computefEq(rho, weight[6], (double2)(dirX.s5, dirY.s5), u);
        fEq5678.z = computefEq(rho, weight[7], (double2)(dirX.s6, dirY.s6), u);
        fEq5678.w = computefEq(rho, weight[8], (double2)(dirX.s7, dirY.s7), u);

        f0 = (1 - omega) * f0 + omega * fEq0;
        f1234 = (1 - omega) * f1234 + omega * fEq1234;
        f5678 = (1 - omega) * f5678 + omega * fEq5678;
    }

    velocityBuffer[pos] = u;

    // Propagate
    // New positions to write (Each thread will write 8 values)
    int8 x8 = id.x;
	int8 y8 = id.y;
	int8 width8 = width;

    int8 nX = x8 + convert_int8(dirX);
    int8 nY = y8 + convert_int8(dirY);
    int8 nPos = nX + width8 * nY;

    // Write center distrivution to thread's location
    of0[pos] = f0;

    int t1 = id.x < get_global_size(0) - 1; // Not on Right boundary
    int t4 = id.y > 0;                      // Not on Upper boundary
    int t3 = id.x > 0;                      // Not on Left boundary
    int t2 = id.y < get_global_size(1) - 1; // Not on lower boundary

    // Propagate to right cell
    if(t1)
        of1234[nPos.s0].x = f1234.x;

    // Propagate to Lower cell
    if(t2)
        of1234[nPos.s1].y = f1234.y;

    // Propagate to left cell
    if(t3)
        of1234[nPos.s2].z = f1234.z;

    // Propagate to Upper cell
    if(t4)
        of1234[nPos.s3].w = f1234.w;

    // Propagate to Lower-Right cell
    if(t1 && t2)
        of5678[nPos.s4].x = f5678.x;

    // Propogate to Lower-Left cell
    if(t2 && t3)
        of5678[nPos.s5].y = f5678.y;

    // Propagate to Upper-Left cell
    if(t3 && t4)
        of5678[nPos.s6].z = f5678.z;

    // Propagate to Upper-Right cell
    if(t4 && t1)
        of5678[nPos.s7].w = f5678.w;
}