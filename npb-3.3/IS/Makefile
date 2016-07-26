SHELL=/bin/sh
BENCHMARK=is
BENCHMARKU=IS

include ../config/make.def

include ../sys/make.common

OBJS = is.o \
       ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/c_wtime.o \
       ${COMMON}/cl_util.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

is.o: is.c  npbparams.h is.h


clean:
	- rm -f *.o *~ mputil* *.bin
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
