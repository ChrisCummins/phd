/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

• Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
• Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#define TOLOWER(x) (('A' <= (x) && (x) <= 'Z') ? ((x - 'A') + 'a') : (x))

/**
* @brief Compare two strings with specified length
* @param text       start position on text string
* @param pattern    start position on pattern string
* @param length     Length to compare
* @return 0-failure, 1-success
*/
int compare(__global const uchar* text, __local const uchar* pattern, uint length)
{
    for(uint l=0; l<length; ++l)
    {
#ifdef CASE_SENSITIVE
        if (text[l] != pattern[l]) return 0;
#else
        if (TOLOWER(text[l]) != pattern[l]) return 0;
#endif
    }
    return 1;
}

/**
* @brief Naive kernel version of string search.
*        Find all pattern positions in the given text
* @param text               Input Text
* @param textLength         Length of the text
* @param pattern            Pattern string
* @param patternLength      Pattern length
* @param resultBuffer       Result of all matched positions
* @param resultCountPerWG   Result counts per Work-Group
* @param maxSearchLength    Maximum search positions for each work-group
* @param localPattern       local buffer for the search pattern
*/
__kernel void
    StringSearchNaive (
      __global uchar* text,
      const uint textLength,
      __global const uchar* pattern,
      const uint patternLength,
      __global int* resultBuffer,
      __global int* resultCountPerWG,
      const uint maxSearchLength,
      __local uchar* localPattern)
{  
    __local volatile uint groupSuccessCounter;

    int localIdx = get_local_id(0);
    int localSize = get_local_size(0);
    int groupIdx = get_group_id(0);

    // Last search idx for all work items
    uint lastSearchIdx = textLength - patternLength + 1;

    // global idx for all work items in a WorkGroup
    uint beginSearchIdx = groupIdx * maxSearchLength;
    uint endSearchIdx = beginSearchIdx + maxSearchLength;
    if(beginSearchIdx > lastSearchIdx) return;
    if(endSearchIdx > lastSearchIdx) endSearchIdx = lastSearchIdx;

    // Copy the pattern from global to local buffer
    for(int idx = localIdx; idx < patternLength; idx+=localSize)
    {
#ifdef CASE_SENSITIVE
        localPattern[idx] = pattern[idx];
#else
        localPattern[idx] = TOLOWER(pattern[idx]);
#endif
    }

    if(localIdx == 0) groupSuccessCounter = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // loop over positions in global buffer
    for(uint stringPos=beginSearchIdx+localIdx; stringPos<endSearchIdx; stringPos+=localSize)
    {
        if (compare(text+stringPos, localPattern, patternLength) == 1)
        {
            int count = atomic_inc(&groupSuccessCounter);
            resultBuffer[beginSearchIdx+count] = stringPos;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if(localIdx == 0) resultCountPerWG[groupIdx] = groupSuccessCounter;
}

/**
* @brief Load-Balance kernel version of string search.
*        Find all pattern positions in the given text
* @param text               Input Text
* @param textLength         Length of the text
* @param pattern            Pattern string
* @param patternLength      Pattern length
* @param resultBuffer       Result of all matched positions
* @param resultCountPerWG   Result counts per Work-Group
* @param maxSearchLength    Maximum search positions for each work-group
* @param localPattern       local buffer for the search pattern
* @param stack1             local stack for store initial 2-byte match 
* @param stack2             local stack for store initial 10-byte match positions
*/
__kernel void
    StringSearchLoadBalance (
      __global uchar* text,
      const uint textLength,
      __global const uchar* pattern,
      const uint patternLength,
      __global int* resultBuffer,
      __global int* resultCountPerWG,
      const uint maxSearchLength,
      __local uchar* localPattern,
      __local int* stack1
#ifdef ENABLE_2ND_LEVEL_FILTER
      , __local int* stack2
#endif
      )
{
    int localIdx = get_local_id(0);
    int localSize = get_local_size(0);
    int groupIdx = get_group_id(0);
        
    __local uint stack1Counter; 
    __local uint stack2Counter;       
    __local uint groupSuccessCounter;
    
    // Initialize the local variaables
    if(localIdx == 0)
    {
        groupSuccessCounter = 0;
        stack1Counter = 0;
        stack2Counter = 0;
    }
    
    // Last search idx for all work items
    uint lastSearchIdx = textLength - patternLength + 1;
    uint stackSize = 0;

    // global idx for all work items in a WorkGroup
    uint beginSearchIdx = groupIdx * maxSearchLength;
    uint endSearchIdx = beginSearchIdx + maxSearchLength;
    if(beginSearchIdx > lastSearchIdx) return;
    if(endSearchIdx > lastSearchIdx) endSearchIdx = lastSearchIdx;
    uint searchLength = endSearchIdx - beginSearchIdx;

    // Copy the pattern from global to local buffer
    for(uint idx = localIdx; idx < patternLength; idx+=localSize)
    {
#ifdef CASE_SENSITIVE
        localPattern[idx] = pattern[idx];
#else
        localPattern[idx] = TOLOWER(pattern[idx]);
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uchar first = localPattern[0];
    uchar second = localPattern[1];
    int stringPos = localIdx;
    int stackPos = 0;
    int revStackPos = 0;

    while (true)    // loop over positions in global buffer
    {

      // Level-1 : Quick filter on 2 char match and store the good positions on stack1.
        if(stringPos < searchLength)
        {
            // Queue the initial match positions. Make sure queue has sufficient positions for each work-item.
#ifdef CASE_SENSITIVE
            if ((first == text[beginSearchIdx+stringPos]) && (second == text[beginSearchIdx+stringPos+1]))
#else
            if ((first == TOLOWER(text[beginSearchIdx+stringPos])) && (second == TOLOWER(text[beginSearchIdx+stringPos+1])))
#endif
            {
                stackPos = atomic_inc(&stack1Counter);
                stack1[stackPos] = stringPos;
            }
        }

        stringPos += localSize;     // next search idx

        barrier(CLK_LOCAL_MEM_FENCE);
            stackSize = stack1Counter;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // continue until stack1 has sufficient good positions for proceed to next Level
        if((stackSize < localSize) && ((((stringPos)/localSize)*localSize) < searchLength)) continue;


#ifdef ENABLE_2ND_LEVEL_FILTER
      // Level-2 : (Processing the stack1 and filling the stack2) For large patterns roll over
      // another 8-bytes from the positions in stack1 and store the match positions in stack2.
        if(localIdx < stackSize)
        {
            revStackPos = atomic_dec(&stack1Counter);
            int pos = stack1[--revStackPos];
#ifdef CASE_SENSITIVE
            bool status = (localPattern[2] == text[beginSearchIdx+pos+2]);
            status = status && (localPattern[3] == text[beginSearchIdx+pos+3]);
            status = status && (localPattern[4] == text[beginSearchIdx+pos+4]);
            status = status && (localPattern[5] == text[beginSearchIdx+pos+5]);
            status = status && (localPattern[6] == text[beginSearchIdx+pos+6]);
            status = status && (localPattern[7] == text[beginSearchIdx+pos+7]);
            status = status && (localPattern[8] == text[beginSearchIdx+pos+8]);
            status = status && (localPattern[9] == text[beginSearchIdx+pos+9]);
#else
            bool status = (localPattern[2] == TOLOWER(text[beginSearchIdx+pos+2]));
            status = status && (localPattern[3] == TOLOWER(text[beginSearchIdx+pos+3]));
            status = status && (localPattern[4] == TOLOWER(text[beginSearchIdx+pos+4]));
            status = status && (localPattern[5] == TOLOWER(text[beginSearchIdx+pos+5]));
            status = status && (localPattern[6] == TOLOWER(text[beginSearchIdx+pos+6]));
            status = status && (localPattern[7] == TOLOWER(text[beginSearchIdx+pos+7]));
            status = status && (localPattern[8] == TOLOWER(text[beginSearchIdx+pos+8]));
            status = status && (localPattern[9] == TOLOWER(text[beginSearchIdx+pos+9]));

#endif
            if (status)
            {
                stackPos = atomic_inc(&stack2Counter);
                stack2[stackPos] = pos;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
            stackSize = stack2Counter;
        barrier(CLK_LOCAL_MEM_FENCE);

        // continue until stack2 has sufficient good positions proceed to next level
        if((stackSize < localSize) && ((((stringPos)/localSize)*localSize) < searchLength)) continue;
#endif


      // Level-3 : (Processing stack1/stack2) Check the remaining positions.
        if(localIdx < stackSize)
        {
#ifdef ENABLE_2ND_LEVEL_FILTER
            revStackPos = atomic_dec(&stack2Counter);
            int pos = stack2[--revStackPos];
            if (compare(text+beginSearchIdx+pos+10, localPattern+10, patternLength-10) == 1)
#else
            revStackPos = atomic_dec(&stack1Counter);
            int pos = stack1[--revStackPos];
            if (compare(text+beginSearchIdx+pos+2, localPattern+2, patternLength-2) == 1)
#endif
            {
                // Full match found
                int count = atomic_inc(&groupSuccessCounter);
                resultBuffer[beginSearchIdx+count] = beginSearchIdx+pos;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if((((stringPos/localSize)*localSize) >= searchLength) && (stack1Counter <= 0) && (stack2Counter <= 0)) break;
    }

    if(localIdx == 0) resultCountPerWG[groupIdx] = groupSuccessCounter;
}
