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

#define S_LOWER_LIMIT 10.0f
#define S_UPPER_LIMIT 100.0f
#define K_LOWER_LIMIT 10.0f
#define K_UPPER_LIMIT 100.0f
#define T_LOWER_LIMIT 1.0f
#define T_UPPER_LIMIT 10.0f
#define R_LOWER_LIMIT 0.01f
#define R_UPPER_LIMIT 0.05f
#define SIGMA_LOWER_LIMIT 0.01f
#define SIGMA_UPPER_LIMIT 0.10f

/**
 * @brief   Abromowitz Stegun approxmimation for PHI (Cumulative Normal Distribution Function)
 * @param   X input value
 * @param   phi pointer to store calculated CND of X
 */
void phi(float4 X, float4* phi)
{
    float4 y;
    float4 absX;
    float4 t;
    float4 result;

    const float4 c1 = (float4)0.319381530f;
    const float4 c2 = (float4)-0.356563782f;
    const float4 c3 = (float4)1.781477937f;
    const float4 c4 = (float4)-1.821255978f;
    const float4 c5 = (float4)1.330274429f;

    const float4 zero = (float4)0.0f;
    const float4 one = (float4)1.0f;
    const float4 two = (float4)2.0f;
    const float4 temp4 = (float4)0.2316419f;

    const float4 oneBySqrt2pi = (float4)0.398942280f;

    absX = fabs(X);
    t = one/(one + temp4 * absX);

    y = one - oneBySqrt2pi * exp(-X*X/two) * t 
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
blackScholes(const __global float4 *randArray,
             int width,
             __global float4 *call,
             __global float4 *put)
{
    float4 d1, d2;
    float4 phiD1, phiD2;
    float4 sigmaSqrtT;
    float4 KexpMinusRT;
    
    size_t xPos = get_global_id(0);
    size_t yPos = get_global_id(1);
    float4 two = (float4)2.0f;
    float4 inRand = randArray[yPos * width + xPos];
    float4 S = S_LOWER_LIMIT * inRand + S_UPPER_LIMIT * (1.0f - inRand);
    float4 K = K_LOWER_LIMIT * inRand + K_UPPER_LIMIT * (1.0f - inRand);
    float4 T = T_LOWER_LIMIT * inRand + T_UPPER_LIMIT * (1.0f - inRand);
    float4 R = R_LOWER_LIMIT * inRand + R_UPPER_LIMIT * (1.0f - inRand);
    float4 sigmaVal = SIGMA_LOWER_LIMIT * inRand + SIGMA_UPPER_LIMIT * (1.0f - inRand);


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

void phi_scalar(float X, float* phi)
{
    float y;
    float absX;
    float t;
    float result;

    const float c1 = (float)0.319381530f;
    const float c2 = (float)-0.356563782f;
    const float c3 = (float)1.781477937f;
    const float c4 = (float)-1.821255978f;
    const float c5 = (float)1.330274429f;

    const float zero = (float)0.0f;
    const float one = (float)1.0f;
    const float two = (float)2.0f;
    const float temp4 = (float)0.2316419f;

    const float oneBySqrt2pi = (float)0.398942280f;

    absX = fabs(X);
    t = one/(one + temp4 * absX);

    y = one - oneBySqrt2pi * exp(-X*X/two) * t 
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
blackScholes_scalar(const __global float *randArray,
             int width,
             __global float *call,
             __global float *put)
{
    float d1, d2;
    float phiD1, phiD2;
    float sigmaSqrtT;
    float KexpMinusRT;
    
    size_t xPos = get_global_id(0);
    size_t yPos = get_global_id(1);
    float two = (float)2.0f;
    float inRand = randArray[yPos * width + xPos];
    float S = S_LOWER_LIMIT * inRand + S_UPPER_LIMIT * (1.0f - inRand);
    float K = K_LOWER_LIMIT * inRand + K_UPPER_LIMIT * (1.0f - inRand);
    float T = T_LOWER_LIMIT * inRand + T_UPPER_LIMIT * (1.0f - inRand);
    float R = R_LOWER_LIMIT * inRand + R_UPPER_LIMIT * (1.0f - inRand);
    float sigmaVal = SIGMA_LOWER_LIMIT * inRand + SIGMA_UPPER_LIMIT * (1.0f - inRand);


    sigmaSqrtT = sigmaVal * sqrt(T);

    d1 = (log(S/K) + (R + sigmaVal * sigmaVal / two)* T)/ sigmaSqrtT;
    d2 = d1 - sigmaSqrtT;

    KexpMinusRT = K * exp(-R * T);
    phi_scalar(d1, &phiD1);
    phi_scalar(d2, &phiD2);
    call[yPos * width + xPos] = S * phiD1 - KexpMinusRT * phiD2;
    phi_scalar(-d1, &phiD1);
    phi_scalar(-d2, &phiD2);
    put[yPos * width + xPos]  = KexpMinusRT * phiD2 - S * phiD1;
}


