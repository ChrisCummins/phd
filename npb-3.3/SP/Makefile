SHELL=/bin/sh
BENCHMARK=sp
BENCHMARKU=SP

include ../config/make.def

include ../sys/make.common

OBJS = sp.o initialize.o exact_rhs.o \
       set_constants.o adi.o rhs.o      \
       x_solve.o ninvr.o y_solve.o pinvr.o    \
       z_solve.o tzetar.o add.o txinvr.o error.o verify.o  \
       ${COMMON}/c_print_results.o ${COMMON}/c_timers.o ${COMMON}/wtime.o \
       ${COMMON}/cl_util.o

# npbparams.h is included by header.h
# The following rule should do the trick but many make programs (not gmake)
# will do the wrong thing and rebuild the world every time (because the
# mod time on header.h is not changed. One solution would be to 
# touch header.h but this might cause confusion if someone has
# accidentally deleted it. Instead, make the dependency on npbparams.h
# explicit in all the lines below (even though dependence is indirect). 

# header.h: npbparams.h

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} $<

sp.o:             sp.c  header.h npbparams.h sp_dim.h
initialize.o:     initialize.c  header.h npbparams.h
exact_rhs.o:      exact_rhs.c  header.h npbparams.h
set_constants.o:  set_constants.c  header.h npbparams.h
adi.o:            adi.c  header.h npbparams.h
rhs.o:            rhs.c  header.h npbparams.h
x_solve.o:        x_solve.c  header.h npbparams.h
ninvr.o:          ninvr.c  header.h npbparams.h
y_solve.o:        y_solve.c  header.h npbparams.h
pinvr.o:          pinvr.c  header.h npbparams.h
z_solve.o:        z_solve.c  header.h npbparams.h
tzetar.o:         tzetar.c  header.h npbparams.h
add.o:            add.c  header.h npbparams.h
txinvr.o:         txinvr.c  header.h npbparams.h
error.o:          error.c  header.h npbparams.h
verify.o:         verify.c  header.h npbparams.h

clean:
	- rm -f *.o *~ mputil* *.bin
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
