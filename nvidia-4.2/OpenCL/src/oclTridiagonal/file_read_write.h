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
 
 /*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 * 
 * Tridiagonal solvers.
 * Test-rig code by UC Davis, Yao Zhang, 2009.
 *
 * NVIDIA, Nikolai Sakharnykh, 2009.
 */

#ifndef _FILE_READ_WRITE_
#define _FILE_READ_WRITE_

void file_read_array(float *x, int system_size, char *file_name)
{
    FILE *fp;
    #ifdef _WIN32
        errno_t err;
        if ((err = fopen_s(&fp, file_name, "rt")) != 0)
    #else
        if ((fp = fopen(file_name, "rt")) == NULL) 
    #endif
        {
	        printf("file open failed.\n");
	        printf("press any key to exit.\n");
	        getchar();
	        exit(1);
        }
	    else
	    {
		    for(int i=0;i<system_size;i++)
            {
                #ifdef _WIN32
                    fscanf_s(fp, "%f", &x[i]);
                #else
                    fscanf(fp, "%f", &x[i]);
                #endif
            }
	    }
	fclose(fp);
}

void file_write_small_systems(float *x,int num_systems,int system_size, char *file_name)
{
    FILE *fp_output;
    #ifdef _WIN32
        errno_t err;
        if ((err = fopen_s(&fp_output, file_name, "wt")) != 0)
    #else
        if ((fp_output = fopen(file_name, "wt")) == NULL) 
    #endif
        {
		    printf("file writing failed.\n");
		    printf("press any key to exit.\n");
		    getchar();
		    exit(1);
        }
	    else
	    {
		    for(int i=0;i<num_systems*system_size;i++)
            {
                if (i%system_size==0) fprintf(fp_output,"***The following is the result of the equation set %d\n",i/system_size );
			    fprintf(fp_output,"%f\n",x[i]);
            }
	    }
	fclose(fp_output);
}

void write_timing_results_1d(double *time,int dim1,char *file_name)
{
    FILE *fp_output;
    #ifdef _WIN32
        errno_t err;
        if ((err = fopen_s(&fp_output, file_name, "wt")) != 0)
    #else
        if ((fp_output = fopen(file_name, "wt")) == NULL) 
    #endif
        {
		    printf("file writing failed.\n");
		    printf("press any key to exit.\n");
		    getchar();
		    exit(1);
        }
	    else
	    {
		    for(int i=0;i<dim1;i++)
            {
                fprintf(fp_output,"%f ",time[i]);
            }

	    }
	fclose(fp_output);
}

void write_timing_results(double time[][16],int dim1,int dim2,char *file_name)
{
    FILE *fp_output;
    #ifdef _WIN32
        errno_t err;
        if ((err = fopen_s(&fp_output, file_name, "wt")) != 0)
    #else
        if ((fp_output = fopen(file_name, "wt")) == NULL) 
    #endif
        {
		    printf("file writing failed.\n");
		    printf("press any key to exit.\n");
		    getchar();
		    exit(1);
        }
	    else
	    {
		    for(int i=0;i<dim1;i++)
            {
		      for(int j=0;j<dim2;j++)
              {
                fprintf(fp_output,"%f ",time[i][j]);
              }
              fprintf(fp_output,"\n");
            }

	    }
	fclose(fp_output);
}

#endif
