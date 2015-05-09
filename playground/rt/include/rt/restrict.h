// -*- c-basic-offset: 8; -*-
#ifndef RT_RESTRICT_H_
#define RT_RESTRICT_H_

// GCC provides a C99 `restrict'-like keyword.
#ifdef __GNUC__
# define restrict __restrict__
#else
# define restrict
#endif

#endif  // RT_RESTRICT_H_
