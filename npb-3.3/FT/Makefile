SHELL=/bin/sh
BENCHMARK=ft
BENCHMARKU=FT

include ../config/make.def

include ../sys/make.common

OBJS = ft.o ${COMMON}/${RAND}.o ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o ${COMMON}/wtime.o \
       ${COMMON}/cl_util.o

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

ft.o: ft.c  global.h npbparams.h ft_dim.h

clean:
	- rm -f *.o *~ mputil* *.bin
	- rm -f ft npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
