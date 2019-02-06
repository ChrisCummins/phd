#ifndef __PRINT_RESULTS_H__
#define __PRINT_RESULTS_H__

void c_print_results( char   *name,
                      char   class,
                      int    n1, 
                      int    n2,
                      int    n3,
                      int    niter,
                      double t,
                      double mops,
		                  char   *optype,
                      int    passed_verification,
                      char   *npbversion,
                      char   *compiletime,
                      char   *cc,
                      char   *clink,
                      char   *c_lib,
                      char   *c_inc,
                      char   *cflags,
                      char   *clinkflags,
                      char   *crand,
                const char   *ocl_dev_type,
                      char   *ocl_dev_name );

#endif //__PRINT_RESULTS_H__
