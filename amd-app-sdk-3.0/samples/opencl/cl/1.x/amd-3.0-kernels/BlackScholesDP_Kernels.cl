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

/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * On invocation of kernel blackScholes, each work thread calculates call price
 * and put price values for given stoke price, option strike price, 
 * time to expiration date, risk free interest and volatility factor.
 */

#ifdef KHR_DP_EXTENSION
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define S_LOWER_LIMIT 10.0
#define S_UPPER_LIMIT 100.0
#define K_LOWER_LIMIT 10.0
#define K_UPPER_LIMIT 100.0
#define T_LOWER_LIMIT 1.0
#define T_UPPER_LIMIT 10.0
#define R_LOWER_LIMIT 0.01
#define R_UPPER_LIMIT 0.05
#define SIGMA_LOWER_LIMIT 0.01
#define SIGMA_UPPER_LIMIT 0.10

/**
 * @brief   Abromowitz Stegun approxmimation for PHI (Cumulative Normal Distribution Function)
 * @param   X input value
 * @param   phi pointer to store calculated CND of X
 */
void phi(double4 X, double4* phi)
{
    double4 y;
    double4 absX;
    double4 t;
    double4 result;

    const double4 c1 = (double4)0.319381530;
    const double4 c2 = (double4)-0.356563782;
    const double4 c3 = (double4)1.781477937;
    const double4 c4 = (double4)-1.821255978;
    const double4 c5 = (double4)1.330274429;

    const double4 zero = (double4)0.0;
    const double4 one = (double4)1.0;
    const double4 two = (double4)2.0;
    const double4 temp4 = (double4)0.2316419;

    const double4 oneBySqrt2pi = (double4)0.398942280;

    absX = fabs(X);
    t = one / (one + temp4 * absX);

    y = one - oneBySqrt2pi * exp(-X * X / two) * t 
        * (c1 + t
              * (c2 + t
                    * (c3 + t
                          * (c4 + t * c5))));

    result = (X < zero)? (one - y) : y;

    *phi = result;
}

/*
 * @brief   Calculates the call and put prices by using Black Scholes model
 * @param   s       Array of random values of current option price
 * @param   sigma   Array of random values sigma
 * @param   k       Array of random values strike price
 * @param   t       Array of random values of expiration time
 * @param   r       Array of random values of risk free interest rate
 * @param   width   Width of call price or put price array
 * @param   call    Array of calculated call price values
 * @param   put     Array of calculated put price values
 */
__kernel 
void
blackScholes(const __global double4 *randArray,
             int width,
             __global double4 *call,
             __global double4 *put)
{
    double4 d1, d2;
    double4 phiD1, phiD2;
    double4 sigmaSqrtT;
    double4 KexpMinusRT;
    
    size_t xPos = get_global_id(0);
    size_t yPos = get_global_id(1);
    double4 two = (double4)2.0;
    double4 inRand = randArray[yPos * width + xPos];
    double4 S = S_LOWER_LIMIT * inRand + S_UPPER_LIMIT * (1.0 - inRand);
    double4 K = K_LOWER_LIMIT * inRand + K_UPPER_LIMIT * (1.0 - inRand);
    double4 T = T_LOWER_LIMIT * inRand + T_UPPER_LIMIT * (1.0 - inRand);
    double4 R = R_LOWER_LIMIT * inRand + R_UPPER_LIMIT * (1.0 - inRand);
    double4 sigmaVal = SIGMA_LOWER_LIMIT * inRand + SIGMA_UPPER_LIMIT * (1.0 - inRand);


    sigmaSqrtT = sigmaVal * sqrt(T);

    d1 = (log(S/K) + (R + sigmaVal * sigmaVal / two)* T)/ sigmaSqrtT;
    d2 = d1 - sigmaSqrtT;

    KexpMinusRT = K * exp(-R * T);
    phi(d1, &phiD1);
    phi(d2, &phiD2);
    call[yPos * width + xPos] = S * phiD1 - KexpMinusRT * phiD2;
    phi(-d1, &phiD1);
    phi(-d2, &phiD2);
    put[yPos * width + xPos]  = KexpMinusRT * phiD2 - S * phiD1;
}

