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
* One invocation of calPriceVega kernel, i.e one work thread caluculates the
* price value and path derivative from given initial price, strike price, 
* interest rate and maturity 
*/
typedef struct _MonteCalroAttribVector
{
    float4 strikePrice;
    float4 c1;
    float4 c2;
    float4 c3;
    float4 initPrice;
    float4 sigma;
    float4 timeStep;
}MonteCarloAttribVector;

constant uint mulFactor = 4;
constant uint stateMask = 1812433253u;
constant uint thirty = 30u;
constant uint thirteen  = 13u;
constant uint fifteen = 15u;
constant uint threeBytes = 8u * 3u;
constant uint mask[4] = {0xfdff37ffu, 0xef7f3f7du, 0xff777b7du, 0x7ff7fb2fu};

constant float one = 1.0f;
constant float intMax = 4294967296.0f;
constant float PI = 3.14159265358979f;
constant float two = 2.0f;

/**
* @brief Left shift
* @param input input to be shifted
* @param output result after shifting input
*/
void 
    lshift128(uint4 input, uint4* output)
{
    unsigned int invshift = 32u - threeBytes;

    uint4 temp;
    temp.x = input.x << threeBytes;
    temp.y = (input.y << threeBytes) | (input.x >> invshift);
    temp.z = (input.z << threeBytes) | (input.y >> invshift);
    temp.w = (input.w << threeBytes) | (input.z >> invshift);

    *output = temp;
}

/**
* @brief Right shift
* @param input input to be shifted
* @param output result after shifting input
*/
void 
    rshift128(uint4 input, uint4* output)
{
    unsigned int invshift = 32u - threeBytes;

    uint4 temp;

    temp.w = input.w >> threeBytes;
    temp.z = (input.z >> threeBytes) | (input.w << invshift);
    temp.y = (input.y >> threeBytes) | (input.z << invshift);
    temp.x = (input.x >> threeBytes) | (input.y << invshift);

    *output = temp;
}

/**
* @brief Generates gaussian random numbers by using 
*        Mersenenne Twister algo and box muller transformation
* @param seed  initial seed value
* @param gaussianRand1 gaussian random number generatred
* @param gaussianRand2 gaussian random number generarted 
* @param nextRand  generated seed for next usage
*/
void 
    generateRand_Vector(uint4 seed,
    float4 *gaussianRand1,
    float4 *gaussianRand2,
    uint4 *nextRand)
{
    uint4 temp[4];

    uint4 state1 = seed;
    uint4 state2 = (uint4)(0); 
    uint4 state3 = (uint4)(0); 
    uint4 state4 = (uint4)(0); 
    uint4 state5 = (uint4)(0);

    uint4 mask4 = (uint4)(stateMask);
    uint4 thirty4 = (uint4)(thirty); 
    uint4 one4 = (uint4)(1u);
    uint4 two4 = (uint4)(2u);
    uint4 three4 = (uint4)(3u);
    uint4 four4 = (uint4)(4u);

    uint4 r1 = (uint4)(0);
    uint4 r2 = (uint4)(0);

    uint4 a = (uint4)(0);
    uint4 b = (uint4)(0); 

    uint4 e = (uint4)(0); 
    uint4 f = (uint4)(0); 

    float4 r; 
    float4 phi;

    float4 temp1;
    float4 temp2;

    //Initializing states.
    state2 = mask4 * (state1 ^ (state1 >> thirty4)) + one4;
    state3 = mask4 * (state2 ^ (state2 >> thirty4)) + two4;
    state4 = mask4 * (state3 ^ (state3 >> thirty4)) + three4;
    state5 = mask4 * (state4 ^ (state4 >> thirty4)) + four4;

    uint i = 0;
    #pragma unroll 4
    for(i = 0; i < mulFactor; ++i)
    {
        switch(i)
        {
        case 0:
            r1 = state4;
            r2 = state5;
            a = state1;
            b = state3;
            break;
        case 1:
            r1 = r2;
            r2 = temp[0];
            a = state2;
            b = state4;
            break;
        case 2:
            r1 = r2;
            r2 = temp[1];
            a = state3;
            b = state5;
            break;
        case 3:
            r1 = r2;
            r2 = temp[2];
            a = state4;
            b = state1;
            break;
        default:
            break;            

        }

        lshift128(a, &e);
        rshift128(r1, &f);

        temp[i].x = a.x ^ e.x ^ ((b.x >> thirteen) & mask[0]) ^ f.x ^ (r2.x << fifteen);
        temp[i].y = a.y ^ e.y ^ ((b.y >> thirteen) & mask[1]) ^ f.y ^ (r2.y << fifteen);
        temp[i].z = a.z ^ e.z ^ ((b.z >> thirteen) & mask[2]) ^ f.z ^ (r2.z << fifteen);
        temp[i].w = a.w ^ e.w ^ ((b.w >> thirteen) & mask[3]) ^ f.w ^ (r2.w << fifteen);
    }        

    temp1 = convert_float4(temp[0]) * one / intMax;
    temp2 = convert_float4(temp[1]) * one / intMax;

    // Applying Box Mullar Transformations.
    r = sqrt((-two) * log(temp1));
    phi  = two * PI * temp2;
    *gaussianRand1 = r * cos(phi);
    *gaussianRand2 = r * sin(phi);
    *nextRand = temp[2];

}

/**
* @brief   calculates the  price and vega for all trajectories
* @param strikePrice	Strike price
* @param meanDeriv1		Average Derive price (from gaussianRand1)
* @param meanDeriv2		Average Derive price (from gaussianRand2)
* @param meanPrice1		Average price (from gaussianRand1)
* @param meanPrice2		Average price (from gaussianRand2)
* @param pathDeriv1		path derive price (for gaussianRand1)
* @param pathDeriv2		path derive price (for gaussianRand2)
* @param price1			price (for gaussianRand1)
* @param price2			price (for gaussianRand2)
*/
void 
    calOutputs_Vector(float4 strikePrice,
    float4 meanDeriv1,
    float4 meanDeriv2, 
    float4 meanPrice1,
    float4 meanPrice2,
    float4 *pathDeriv1,
    float4 *pathDeriv2, 
    float4 *priceVec1,
    float4 *priceVec2)
{
    float4 temp1 = (float4)0.0f;
    float4 temp2 = (float4)0.0f;
    float4 temp3 = (float4)0.0f;
    float4 temp4 = (float4)0.0f;

    float4 tempDiff1 = meanPrice1 - strikePrice;
    float4 tempDiff2 = meanPrice2 - strikePrice;
    if(tempDiff1.x > 0.0f)
    {
        temp1.x = 1.0f;
        temp3.x = tempDiff1.x;
    }
    if(tempDiff1.y > 0.0f)
    {
        temp1.y = 1.0f;
        temp3.y = tempDiff1.y ;
    }
    if(tempDiff1.z > 0.0f)
    {
        temp1.z = 1.0f;
        temp3.z = tempDiff1.z;
    }
    if(tempDiff1.w > 0.0f)
    {
        temp1.w = 1.0f;
        temp3.w = tempDiff1.w;
    }

    if(tempDiff2.x > 0.0f)
    {
        temp2.x = 1.0f;
        temp4.x = tempDiff2.x;
    }
    if(tempDiff2.y > 0.0f)
    {
        temp2.y = 1.0f;
        temp4.y = tempDiff2.y;
    }
    if(tempDiff2.z > 0.0f)
    {
        temp2.z = 1.0f;
        temp4.z = tempDiff2.z;
    }
    if(tempDiff2.w > 0.0f)
    {
        temp2.w = 1.0f;
        temp4.w = tempDiff2.w;
    }

    *pathDeriv1 = meanDeriv1 * temp1; 
    *pathDeriv2 = meanDeriv2 * temp2; 
    *priceVec1 = temp3; 
    *priceVec2 = temp4;	
}


/**
* @brief   Calculates the  price and vega for all trajectories for given random numbers(For Vector Kernel)
* @param   attrib  structure of inputs for simulation
* @param   priceSamples    array of calculated price samples
* @param   pathDeriv   array calculated path derivatives 
* @param   sData1   array used for blockwise reduction
*/
__kernel 
    void
    calPriceVega_Vector(MonteCarloAttribVector attrib,
    int noOfSum,
    __global uint4 *randArray,
    __global float4 *priceSamples,
    __global float4 *pathDeriv,
    __local float8 *sData1)
{
    float4 strikePrice = attrib.strikePrice;
    float4 c1 = attrib.c1;
    float4 c2 = attrib.c2;
    float4 c3 = attrib.c3;
    float4 initPrice = attrib.initPrice;
    float4 sigma = attrib.sigma;
    float4 timeStep = attrib.timeStep;

    int2 localIdx;
    int2 groupIdx;
    int2 groupDim;
    localIdx.x = (int)get_local_id(0);
    localIdx.y = (int)get_local_id(1);
    groupIdx.x = (int)get_group_id(0);
    groupIdx.y = (int)get_group_id(1);
    groupDim.x = (int)get_local_size(0);
    groupDim.y = (int)get_local_size(1);

    int xDim = (int)get_global_size(0);
    int yDim = (int)get_global_size(1);
    int xPos = (int)get_global_id(0);
    int yPos = (int)get_global_id(1);
    int gidx=yPos * xDim + xPos;
    int bidx=groupIdx.y*xDim/groupDim.x+groupIdx.x;
    int lidx=localIdx.y*groupDim.x + localIdx.x;
    
    float4 temp = (float4)0.0f;
    float4 temp1 = (float4)0.0f;
    float4 temp2 = (float4)0.0f;

    float4 price1 = (float4)0.0f;
    float4 price2 = (float4)0.0f;
    float4 pathDeriv1 = (float4)0.0f;
    float4 pathDeriv2 = (float4)0.0f;

    float4 trajPrice1 = initPrice;
    float4 trajPrice2 = initPrice;

    float4 sumPrice1 = initPrice;
    float4 sumPrice2 = initPrice;

    float4 meanPrice1 = temp;
    float4 meanPrice2 = temp;

    float4 sumDeriv1 = temp;
    float4 sumDeriv2 = temp;

    float4 meanDeriv1 = temp;
    float4 meanDeriv2 = temp;

    float4 finalRandf1 = temp;
    float4 finalRandf2 = temp;

    uint4 nextRand = randArray[gidx];

    //Run the Monte Carlo simulation a total of Num_Sum - 1 times
    for(int i = 1; i < noOfSum; i++)
    {
        uint4 tempRand = nextRand;
        generateRand_Vector(tempRand, &finalRandf1, &finalRandf2, &nextRand);

        //Calculate the trajectory price and sum price for all trajectories

        temp1 += c1 + c2 * finalRandf1;
        temp2 += c1 + c2 * finalRandf2;
        trajPrice1 = trajPrice1 * exp(c1 + c2 * finalRandf1);
        trajPrice2 = trajPrice2 * exp(c1 + c2 * finalRandf2);

        sumPrice1 = sumPrice1 + trajPrice1;
        sumPrice2 = sumPrice2 + trajPrice2;

        temp = c3 * timeStep * i;

        // Calculate the derivative price for all trajectories
        sumDeriv1 = sumDeriv1 + trajPrice1 
            * (temp1 - temp) / sigma;

        sumDeriv2 = sumDeriv2 + trajPrice2 
            * (temp2 - temp) / sigma;
    }

    //Calculate the average price and average derivative of each simulated path
    meanPrice1 = sumPrice1 / noOfSum;
    meanPrice2 = sumPrice2 / noOfSum;
    meanDeriv1 = sumDeriv1 / noOfSum;
    meanDeriv2 = sumDeriv2 / noOfSum;

    calOutputs_Vector(strikePrice, meanDeriv1, meanDeriv2, meanPrice1,
        meanPrice2, &pathDeriv1, &pathDeriv2, &price1, &price2);

     //Do the reduction blockwise and store the result in sData1[0]
     sData1[lidx]=(float8)(price1+price2,pathDeriv1+pathDeriv2);
     barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int s = (groupDim.x*groupDim.y)>> 1; s > 0; s >>= 1) 
    {
        if(lidx < s) 
        {
            sData1[lidx] += sData1[lidx + s];
        }
          barrier(CLK_LOCAL_MEM_FENCE);	
    }

    // Write the reduction result of  this block to global memory
    if ((localIdx.y==0) && (localIdx.x==0) ){
    priceSamples[bidx] = sData1[0].lo;
    pathDeriv[bidx] = sData1[0].hi;
    
    }
}

typedef struct _MonteCalroAttribScalar
{
    float strikePrice;
    float c1;
    float c2;
    float c3;
    float initPrice;
    float sigma;
    float timeStep;
}MonteCarloAttribScalar;

/**
* @brief Generates gaussian random numbers by using Mersenenne
*        Twister algo and box muller transformation(For Scalar Kernel)
* @param randPrice		local memory used for price randon num generation
* @param randVega		local memory used for vega randon num generation
* @param seed			Initial seed value for random number generation
* @param gaussianRand1	gaussian random number generatred
* @param gaussianRand2	gaussian random number generarted 
* @param nextRand		generated seed for next usage
* @param lidx			local thread index
*/
void 
    generateRand_Scalar(
    __local uint* randPrice,
    __local uint* randVega,
    uint seed,
    float *gaussianRand1,
    float *gaussianRand2,
    uint *nextRand,
    uint lidx)
{
    uint r2 = 0;
    uint b = 0; 

    uint state[4]; 
    uint temp[4];

    //Initializing states.
    state[0] = stateMask * (seed	 ^ (seed	 >> thirty)) + 1u;
    state[1] = stateMask * (state[0] ^ (state[0] >> thirty)) + 2u;
    state[2] = stateMask * (state[1] ^ (state[1] >> thirty)) + 3u;
    state[3] = stateMask * (state[2] ^ (state[2] >> thirty)) + 4u;

    #pragma unroll 4
    for(uint i = 0; i < mulFactor; ++i)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        randPrice[lidx] = ((i==0) ? state[2] : r2);
        r2 = ((i==0) ? state[3] : temp[i-1]);
        randVega[lidx] = ((i==0) ? seed : state[i-1]);
        b = ((i==3) ? seed : state[i+1]);

        barrier(CLK_LOCAL_MEM_FENCE);	

        // lshift128
        uint e = randVega[lidx] << threeBytes;
        if (lidx % 4) {
            e |= (randVega[lidx-1] >> (32u - threeBytes));
        }

        // rshift128
        uint f = randPrice[lidx] >> threeBytes;
        if ((lidx+1) % 4) {
            f |= (randPrice[lidx+1] << (32u - threeBytes));
        }

        temp[i] = randVega[lidx] ^ e ^ ((b >> thirteen) & mask[lidx%4]) ^ f ^ (r2 << fifteen);
    }        

    // Applying Box Mullar Transformations.
    float r = sqrt((-two) * log(temp[0] * one / intMax));
    float phi  = two * PI * temp[1] * one / intMax;
    *gaussianRand1 = r * cos(phi);
    *gaussianRand2 = r * sin(phi);
    *nextRand = temp[2];
}

/**
* @brief   calculates the  price and vega for all trajectories(For Scalar Kernel)
* @param strikePrice	Strike price
* @param meanDeriv1		Average Derive price (from gaussianRand1)
* @param meanDeriv2		Average Derive price (from gaussianRand2)
* @param meanPrice1		Average price (from gaussianRand1)
* @param meanPrice2		Average price (from gaussianRand2)
* @param pathDeriv1		path derive price (for gaussianRand1)
* @param pathDeriv2		path derive price (for gaussianRand2)
* @param price1			price (for gaussianRand1)
* @param price2			price (for gaussianRand2)
*/
void 
    calOutputs_Scalar(float strikePrice,
    float meanDeriv1,
    float meanDeriv2, 
    float meanPrice1,
    float meanPrice2,
    float *pathDeriv1,
    float *pathDeriv2, 
    float *price1,
    float *price2)
{
    *price1 = 0.0f;
    *price2 = 0.0f;
    *pathDeriv1 = 0.0f;
    *pathDeriv2 = 0.0f;
    float tempDiff1 = meanPrice1 - strikePrice;
    float tempDiff2 = meanPrice2 - strikePrice;
    if(tempDiff1 > 0.0f)
    {
        *pathDeriv1 = meanDeriv1;
        *price1 = tempDiff1;
    }
    if(tempDiff2 > 0.0f)
    {
        *pathDeriv2 = meanDeriv2;
        *price2 = tempDiff2;
    }
}


/**
* @brief   Calculates the  price and vega for all trajectories for given random numbers(For Scalar Kernel)
* @param   attrib		structure of inputs for simulation
* @param   noOfSum		Number of Monte Carlo simulation
* @param   priceSamples array of calculated price samples
* @param   pathDeriv    array calculated path derivatives 
* @param   sData1		array used for blockwise reduction
* @param   randPrice	local memory used for price randon num generation
* @param   randVega		local memory used for vega randon num generation
*/
__kernel 
    void 
    calPriceVega_Scalar(MonteCarloAttribScalar attrib,
        int noOfSum,
        __global uint *randArray,
        __global float *priceSamples,
        __global float *pathDeriv,
        __local float2 *sData1,
        __local uint *randPrice,
        __local uint *randVega)
{
    float strikePrice = attrib.strikePrice;
    float c1 = attrib.c1;
    float c2 = attrib.c2;
    float c3 = attrib.c3;
    float initPrice = attrib.initPrice;
    float sigma = attrib.sigma;
    float timeStep = attrib.timeStep;

    int2 localIdx;
    int2 groupIdx;
    int2 groupDim;
    localIdx.x = (int)get_local_id(0);
    localIdx.y = (int)get_local_id(1);
    groupIdx.x = (int)get_group_id(0);
    groupIdx.y = (int)get_group_id(1);
    groupDim.x = (int)get_local_size(0);
    groupDim.y = (int)get_local_size(1);

    int xDim = (int)get_global_size(0);
    int yDim = (int)get_global_size(1);
    int xPos = (int)get_global_id(0);
    int yPos = (int)get_global_id(1);
    int gidx=yPos * xDim + xPos;
    int bidx=groupIdx.y*xDim/groupDim.x+groupIdx.x;
    int lidx=localIdx.y*groupDim.x + localIdx.x;

    float temp  = 0.0f;
    float temp1 = 0.0f;
    float temp2 = 0.0f;

    float price1 = 0.0f;
    float price2 = 0.0f;
    float pathDeriv1 = 0.0f;
    float pathDeriv2 = 0.0f;

    float trajPrice1 = initPrice;
    float trajPrice2 = initPrice;

    float sumPrice1 = initPrice;
    float sumPrice2 = initPrice;

    float meanPrice1 = 0.0f;
    float meanPrice2 = 0.0f;

    float sumDeriv1 = 0.0f;
    float sumDeriv2 = 0.0f;

    float meanDeriv1 = 0.0f;
    float meanDeriv2 = 0.0f;

    float finalRandf1 = 0.0f;
    float finalRandf2 = 0.0f;

    uint nextRand = randArray[gidx];

    //Run the Monte Carlo simulation a total of Num_Sum - 1 times
    for(int i = 1; i < noOfSum; i++)
    {
        uint tempRand = nextRand;
        generateRand_Scalar(randPrice, randVega, tempRand, &finalRandf1, &finalRandf2, &nextRand, lidx);

        //Calculate the trajectory price and sum price for all trajectories
        temp1 += c1 + c2 * finalRandf1;
        temp2 += c1 + c2 * finalRandf2;
        trajPrice1 = trajPrice1 * exp(c1 + c2 * finalRandf1);
        trajPrice2 = trajPrice2 * exp(c1 + c2 * finalRandf2);

        sumPrice1 = sumPrice1 + trajPrice1;
        sumPrice2 = sumPrice2 + trajPrice2;

        temp = c3 * timeStep * i;

        // Calculate the derivative price for all trajectories
        sumDeriv1 = sumDeriv1 + trajPrice1 
            * (temp1 - temp) / sigma;

        sumDeriv2 = sumDeriv2 + trajPrice2 
            * (temp2 - temp) / sigma;
    }

    // Calculate the average price and average derivative of each simulated path
    meanPrice1 = sumPrice1 / noOfSum;
    meanPrice2 = sumPrice2 / noOfSum;
    meanDeriv1 = sumDeriv1 / noOfSum;
    meanDeriv2 = sumDeriv2 / noOfSum;

    calOutputs_Scalar(strikePrice, meanDeriv1, meanDeriv2, meanPrice1,
                     meanPrice2, &pathDeriv1, &pathDeriv2, &price1, &price2);

    // Do the reduction blockwise and store the result in sData1[0-7]
    sData1[lidx] = (float2)(price1+price2, pathDeriv1+pathDeriv2);
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int s = (groupDim.x*groupDim.y)>> 1; s >= 4; s >>= 1) 
    {
        if(lidx < s) 
        {
            sData1[lidx] += sData1[lidx + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);	
    }

    // Write the reduction result of  this block to global memory
    if (lidx < 4){
        priceSamples[bidx*4+lidx] = sData1[lidx].lo;
        pathDeriv[bidx*4+lidx] = sData1[lidx].hi;
    }
}