/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// standard utilities and systems includes
#include <oclUtils.h>

//*****************************************************************
//! Exported Host/C++ RGB 3x3 Median function
//! Gradient intensity is from RSS combination of H and V gradient components
//! R, G and B medians are treated separately 
//!
//! @param uiInputImage     pointer to input data
//! @param uiOutputImage    pointer to output dataa
//! @param uiWidth          width of image
//! @param uiHeight         height of image
//*****************************************************************
extern "C" double MedianFilterHost(unsigned int* uiInputImage, unsigned int* uiOutputImage, 
                                   unsigned int uiWidth, unsigned int uiHeight)
{
    // start computation timer
    shrDeltaT(0);

	// do the Median 
	for(unsigned int y = 0; y < uiHeight; y++)			// all the rows
	{
		for(unsigned int x = 0; x < uiWidth; x++)		// all the columns
		{
            // local registers for working with RGB subpixels and managing border
            unsigned char* ucRGBA; 
            const unsigned int uiZero = 0U;

		    // reset accumulators  
            float fMedianEstimate [3] = {128.0f, 128.0f, 128.0f};
            float fMinBound [3]= {0.0f, 0.0f, 0.0f};
            float fMaxBound[3] = {255.0f, 255.0f, 255.0f};

		    // now find the median using a binary search - Divide and Conquer 256 gv levels for 8 bit plane
		    for(int iSearch = 0; iSearch < 8; iSearch++)  
		    {
                unsigned int uiHighCount[3] = {0,0,0};

			    for (int iRow = -1; iRow <= 1 ; iRow++)
			    {
                    int iLocalOffset = (int)((iRow + y) * uiWidth) + x - 1;

				    // Left Pix (RGB)
                    // Read in pixel value to local register:  if boundary pixel, use zero
                    if ((x > 0) && ((y + iRow) >= 0) && ((y + iRow) < uiHeight))
                    {
                        ucRGBA = (unsigned char*)&uiInputImage [iLocalOffset];
                    }
                    else 
                    {
                        ucRGBA = (unsigned char*)&uiZero;
                    }
				    uiHighCount[0] += (fMedianEstimate[0] < ucRGBA[0]);					
				    uiHighCount[1] += (fMedianEstimate[1] < ucRGBA[1]);					
				    uiHighCount[2] += (fMedianEstimate[2] < ucRGBA[2]);	

				    // Middle Pix (RGB)
                    // Increment offset and read in next pixel value to a local register:  if boundary pixel, use zero
                    iLocalOffset++;
                    if (((y + iRow) >= 0) && ((y + iRow) < uiHeight)) 
                    {
                        ucRGBA = (unsigned char*)&uiInputImage [iLocalOffset];
                    }
                    else 
                    {
                        ucRGBA = (unsigned char*)&uiZero;
                    }
				    uiHighCount[0] += (fMedianEstimate[0] < ucRGBA[0]);					
				    uiHighCount[1] += (fMedianEstimate[1] < ucRGBA[1]);					
				    uiHighCount[2] += (fMedianEstimate[2] < ucRGBA[2]);	

				    // Right Pix (RGB)
                    // Increment offset and read in next pixel value to a local register:  if boundary pixel, use zero
                    iLocalOffset++;
                    if ((x < (uiWidth - 1)) && ((y + iRow) >= 0) && ((y + iRow) < uiHeight))
                    {
                        ucRGBA = (unsigned char*)&uiInputImage [iLocalOffset];
                    }
                    else 
                    {
                        ucRGBA = (unsigned char*)&uiZero;
                    }
				    uiHighCount[0] += (fMedianEstimate[0] < ucRGBA[0]);					
				    uiHighCount[1] += (fMedianEstimate[1] < ucRGBA[1]);					
				    uiHighCount[2] += (fMedianEstimate[2] < ucRGBA[2]);	
			    }

			    //********************************
			    // reset the appropriate bound, depending upon counter
			    if(uiHighCount[0] > 4)
			    {
				    fMinBound[0] = fMedianEstimate[0];				
			    }
			    else
			    {
				    fMaxBound[0] = fMedianEstimate[0];				
			    }

			    if(uiHighCount[1] > 4)
			    {
				    fMinBound[1] = fMedianEstimate[1];				
			    }
			    else
			    {
				    fMaxBound[1] = fMedianEstimate[1];				
			    }

			    if(uiHighCount[2] > 4)
			    {
				    fMinBound[2] = fMedianEstimate[2];				
			    }
			    else
			    {
				    fMaxBound[2] = fMedianEstimate[2];				
			    }

			    // refine the estimate
			    fMedianEstimate[0] = 0.5f * (fMaxBound[0] + fMinBound[0]);
			    fMedianEstimate[1] = 0.5f * (fMaxBound[1] + fMinBound[1]);
			    fMedianEstimate[2] = 0.5f * (fMaxBound[2] + fMinBound[2]);
		    }

            // pack into a monochrome uint 
            unsigned int uiPackedPix = 0x000000FF & (unsigned int)(fMedianEstimate[0] + 0.5f);
            uiPackedPix |= 0x0000FF00 & (((unsigned int)(fMedianEstimate[1] + 0.5f)) << 8);
            uiPackedPix |= 0x00FF0000 & (((unsigned int)(fMedianEstimate[2] + 0.5f)) << 16);

			// copy to output
			uiOutputImage[y * uiWidth + x] = uiPackedPix;	
		}
	}

    // return computation elapsed time in seconds
    return shrDeltaT(0);
}
