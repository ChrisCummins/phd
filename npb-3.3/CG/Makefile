SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG

include ../config/make.def

include ../sys/make.common

OBJS = cg.o \
        ${COMMON}/c_print_results.o  \
        ${COMMON}/${RAND}.o \
        ${COMMON}/c_timers.o \
        ${COMMON}/wtime.o \
        ${COMMON}/cl_util.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

cg.o:		cg.c  globals.h npbparams.h

clean:
	- rm -f *.o *~ *.bin
	- rm -f npbparams.h core
