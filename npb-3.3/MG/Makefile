SHELL=/bin/sh
BENCHMARK=mg
BENCHMARKU=MG

include ../config/make.def

include ../sys/make.common

OBJS = mg.o \
       ${COMMON}/c_print_results.o  \
       ${COMMON}/${RAND}.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/wtime.o \
	${COMMON}/cl_util.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

mg.o:		mg.c globals.h npbparams.h mg_dim.h

clean:
	- rm -f *.o *~ 
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
