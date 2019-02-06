//========================================================================================================================================================================================================200
//======================================================================================================================================================150
//====================================================================================================100
//==================================================50

//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdlib.h>								// (in directory known to compiler)
#include <math.h>								// (in directory known to compiler)
#include <string.h>								// (in directory known to compiler)

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"								// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./util/avi/avilib.h"					// (in directory)							needed by avi functions
#include "./util/avi/avimod.h"					// (in directory)							needed by avi functions

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel/kernel_gpu_opencl_wrapper.h"

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

int 
main(	int argc, 
		char* argv []){


  printf("WG size of kernel = %d \n", NUMBER_THREADS);
	//======================================================================================================================================================150
	//	VARIABLES
	//======================================================================================================================================================150

	// time
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;

	// other
	int i;
	avi_t* frames;

	time0 = get_time();

	//======================================================================================================================================================150
	//	STRUCTURES, GLOBAL STRUCTURE VARIABLES
	//======================================================================================================================================================150

	params_common common;
	common.common_mem = sizeof(params_common);

	//======================================================================================================================================================150
	// 	FRAME INFO
	//======================================================================================================================================================150

	// variables
	char* video_file_name;

	// open movie file
 	video_file_name = (char *) "../../../../../rodinia_data/heartwall/test.avi";
	frames = (avi_t*)AVI_open_input_file(video_file_name, 1);														// added casting
	if (frames == NULL)  {
		   AVI_print_error((char *) "Error with AVI_open_input_file");
		   return -1;
	}

	// dimensions
	common.no_frames = AVI_video_frames(frames);
	common.frame_rows = AVI_video_height(frames);
	common.frame_cols = AVI_video_width(frames);
	common.frame_elem = common.frame_rows * common.frame_cols;
	common.frame_mem = sizeof(fp) * common.frame_elem;

	time1 = get_time();

	//======================================================================================================================================================150
	// 	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================150

	if(argc!=2){
		printf("ERROR: missing argument (number of frames to processed) or too many arguments\n");
		return 0;
	}
	else{
		common.frames_processed = atoi(argv[1]);
		if(common.frames_processed<0 || common.frames_processed>common.no_frames){
			printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n", common.frames_processed, common.no_frames);
			return 0;
		}
	}

	time2 = get_time();

	//======================================================================================================================================================150
	//	INPUTS
	//======================================================================================================================================================150

	//====================================================================================================100
	//	READ PARAMETERS FROM FILE
	//====================================================================================================100

	read_parameters(	"../../../../../rodinia_data/heartwall/input.txt",
						&common.tSize,
						&common.sSize,
						&common.maxMove,
						&common.alpha);

	//====================================================================================================100
	//	READ SIZE OF INPUTS FROM FILE
	//====================================================================================================100

	read_header(	"../../../../../rodinia_data/heartwall/input.txt",
					&common.endoPoints,
					&common.epiPoints);

	common.allPoints = common.endoPoints + common.epiPoints;

	//====================================================================================================100
	//	READ DATA FROM FILE
	//====================================================================================================100

	//==================================================50
	//	ENDO POINTS MEMORY ALLOCATION
	//==================================================50

	common.endo_mem = sizeof(int) * common.endoPoints;

	int* endoRow;
	endoRow = (int*)malloc(common.endo_mem);
	int* endoCol;
	endoCol = (int*)malloc(common.endo_mem);
	int* tEndoRowLoc;
	tEndoRowLoc = (int*)malloc(common.endo_mem * common.no_frames);
	int* tEndoColLoc;
	tEndoColLoc = (int*)malloc(common.endo_mem * common.no_frames);

	//==================================================50
	//	EPI POINTS MEMORY ALLOCATION
	//==================================================50

	common.epi_mem = sizeof(int) * common.epiPoints;

	int* epiRow;
	epiRow = (int *)malloc(common.epi_mem);
	int* epiCol;
	epiCol = (int *)malloc(common.epi_mem);
	int* tEpiRowLoc;
	tEpiRowLoc = (int *)malloc(common.epi_mem * common.no_frames);
	int* tEpiColLoc;
	tEpiColLoc = (int *)malloc(common.epi_mem * common.no_frames);

	//==================================================50
	//	READ DATA FROM FILE
	//==================================================50

	read_data(	"../../../../../rodinia_data/heartwall/input.txt",
				common.endoPoints,
				endoRow,
				endoCol,
				common.epiPoints,
				epiRow,
				epiCol);

	//==================================================50
	//	End
	//==================================================50

	//====================================================================================================100
	//	End
	//====================================================================================================100

	time3 = get_time();

	//======================================================================================================================================================150
	//	KERNELL WRAPPER CALL
	//======================================================================================================================================================150

	kernel_gpu_opencl_wrapper(	common,
								endoRow,
								endoCol,
								tEndoRowLoc,
								tEndoColLoc,
								epiRow,
								epiCol,
								tEpiRowLoc,
								tEpiColLoc,
								frames);

	time4 = get_time();

	//==================================================50
	//	DUMP DATA TO FILE
	//==================================================50
#ifdef OUTPUT
	write_data(	"result.txt",
			common.no_frames,
			common.frames_processed,		
				common.endoPoints,
				tEndoRowLoc,
				tEndoColLoc,
				common.epiPoints,
				tEpiRowLoc,
				tEpiColLoc);

#endif
	//==================================================50
	//	End
	//==================================================50


	//======================================================================================================================================================150
	//	DEALLOCATION
	//======================================================================================================================================================150

	//====================================================================================================100
	// endo points
	//====================================================================================================100

	free(endoRow);
	free(endoCol);
	free(tEndoRowLoc);
	free(tEndoColLoc);

	//====================================================================================================100
	// epi points
	//====================================================================================================100

	free(epiRow);
	free(epiCol);
	free(tEpiRowLoc);
	free(tEpiColLoc);

	//====================================================================================================100
	//	End
	//====================================================================================================100

	time5= get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of the application:\n");
	printf("%15.12f s, %15.12f % : READ INITIAL VIDEO FRAME\n",												(fp) (time1-time0) / 1000000, (fp) (time1-time0) / (fp) (time5-time0) * 100);
	printf("%15.12f s, %15.12f % : READ COMMAND LINE PARAMETERS\n",											(fp) (time2-time1) / 1000000, (fp) (time2-time1) / (fp) (time5-time0) * 100);
	printf("%15.12f s, %15.12f % : READ INPUTS FROM FILE\n",												(fp) (time3-time2) / 1000000, (fp) (time3-time2) / (fp) (time5-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU ALLOCATION, COPYING, COMPUTATION\n",									(fp) (time4-time3) / 1000000, (fp) (time4-time3) / (fp) (time5-time0) * 100);
	printf("%15.12f s, %15.12f % : FREE MEMORY\n", 															(fp) (time5-time4) / 1000000, (fp) (time5-time4) / (fp) (time5-time0) * 100);
	printf("Total time:\n");
	printf("%15.12f s\n", 																					(fp) (time5-time0) / 1000000);

	//======================================================================================================================================================150
	//	End
	//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200

}
