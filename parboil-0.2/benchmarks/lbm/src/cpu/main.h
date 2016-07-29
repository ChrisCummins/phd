/* $Id: main.h,v 1.1 2008/03/04 17:30:03 stratton Exp $ */

/*############################################################################*/

#ifndef _MAIN_H_
#define _MAIN_H_

/*############################################################################*/

#include "config.h"
#include "parboil.h"

#if !defined(SPEC_CPU)
#include <sys/times.h>
#endif

/*############################################################################*/

#if !defined(SPEC_CPU)
typedef struct {
	float timeScale;
	clock_t tickStart, tickStop;
	struct tms timeStart, timeStop;

} MAIN_Time;
#endif

typedef enum {NOTHING = 0, COMPARE, STORE} MAIN_Action;
typedef enum {LDC = 0, CHANNEL} MAIN_SimType;

typedef struct {
	int nTimeSteps;
	char* resultFilename;
	MAIN_Action action;
	MAIN_SimType simType;
	char* obstacleFilename;
} MAIN_Param;

/*############################################################################*/

void MAIN_parseCommandLine( int nArgs, char* arg[], MAIN_Param* param, struct pb_Parameters* );
void MAIN_printInfo( const MAIN_Param* param );
void MAIN_initialize( const MAIN_Param* param );
void MAIN_finalize( const MAIN_Param* param );

#if !defined(SPEC_CPU)
void MAIN_startClock( MAIN_Time* time );
void MAIN_stopClock( MAIN_Time* time, const MAIN_Param* param );
#endif

/*############################################################################*/

#endif /* _MAIN_H_ */
