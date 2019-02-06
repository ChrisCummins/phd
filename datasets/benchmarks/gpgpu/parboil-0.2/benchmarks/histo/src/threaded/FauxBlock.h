/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#ifndef FAUX_BLOCK_H__
#define FAUX_BLOCK_H__

#include <string.h>
#include <stdlib.h>
#include <pthread.h>

typedef void (*void_func_t) (void  *);

#define FAUX_BLOCK_MAX_INT_ARGS (64)

#define FAUX_BLOCK_FUNC 

typedef struct faux_block
{
    void_func_t func;
    void*       tid;
    double      args[FAUX_BLOCK_MAX_INT_ARGS/2];     // Use an array of doubles to ensure most restrictive memory alignment
} faux_block_t;

faux_block_t make_faux_block          (void_func_t func, const void *top, const void *bottom, size_t last_size);
void         exec_faux_block          (faux_block_t *block);
int          exec_faux_block_deferred (faux_block_t block[], size_t len);
void         faux_block_exec_join     (faux_block_t block[], size_t len);
void         faux_block_run           (faux_block_t block[], size_t len);

void         clear_runners            ();

void not_enough_arg_space(void *arg);

#define MAKE_FUNC_HELPER(name__)\
        if (sizeof(s_) > sizeof(blk.args))\
        {\
            blk.func = not_enough_arg_space;\
            blk.tid  = (void*)-1;\
        }\
        else\
        {\
            blk.func = name__ ## _struct_func;  \
            blk.tid  = 0;\
            memcpy(blk.args, &s_, sizeof(s_));\
        }



#define MAKE_FUNC_0_ARGS(scope__, name__)\
    static void FAUX_BLOCK_FUNC name__ ## _func();\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func();\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block ()\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func()



#define MAKE_FUNC_1_ARGS(scope__, name__, t01_, v01_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_)



#define MAKE_FUNC_2_ARGS(scope__, name__, t01_, v01_, t02_, v02_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_)


#define MAKE_FUNC_3_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_)


#define MAKE_FUNC_4_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_)



#define MAKE_FUNC_5_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_)

#define MAKE_FUNC_6_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_, t06_, v06_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
        t06_ v06_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_,\
                        arg->v06_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_,\
                                                                    t06_ v06_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_,\
                                                 v06_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_)


#define MAKE_FUNC_7_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_, t06_, v06_,\
                                          t07_, v07_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
        t06_ v06_;\
        t07_ v07_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_,\
                        arg->v06_,\
                        arg->v07_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_,\
                                                                    t06_ v06_,\
                                                                    t07_ v07_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_,\
                                                 v06_,\
                                                 v07_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_)


#define MAKE_FUNC_8_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_, t06_, v06_,\
                                          t07_, v07_, t08_, v08_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
        t06_ v06_;\
        t07_ v07_;\
        t08_ v08_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_,\
                        arg->v06_,\
                        arg->v07_,\
                        arg->v08_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_,\
                                                                    t06_ v06_,\
                                                                    t07_ v07_,\
                                                                    t08_ v08_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_,\
                                                 v06_,\
                                                 v07_,\
                                                 v08_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_)


#define MAKE_FUNC_9_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_, t06_, v06_,\
                                          t07_, v07_, t08_, v08_, t09_, v09_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
        t06_ v06_;\
        t07_ v07_;\
        t08_ v08_;\
        t09_ v09_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_,\
                        arg->v06_,\
                        arg->v07_,\
                        arg->v08_,\
                        arg->v09_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_,\
                                                                    t06_ v06_,\
                                                                    t07_ v07_,\
                                                                    t08_ v08_,\
                                                                    t09_ v09_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_,\
                                                 v06_,\
                                                 v07_,\
                                                 v08_,\
                                                 v09_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_)


#define MAKE_FUNC_10_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_, t06_, v06_,\
                                           t07_, v07_, t08_, v08_, t09_, v09_, t10_, v10_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
        t06_ v06_;\
        t07_ v07_;\
        t08_ v08_;\
        t09_ v09_;\
        t10_ v10_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_,\
                        arg->v06_,\
                        arg->v07_,\
                        arg->v08_,\
                        arg->v09_,\
                        arg->v10_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_,\
                                                                    t06_ v06_,\
                                                                    t07_ v07_,\
                                                                    t08_ v08_,\
                                                                    t09_ v09_,\
                                                                    t10_ v10_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_,\
                                                 v06_,\
                                                 v07_,\
                                                 v08_,\
                                                 v09_,\
                                                 v10_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_)


#define MAKE_FUNC_11_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_, t06_, v06_,\
                                           t07_, v07_, t08_, v08_, t09_, v09_, t10_, v10_, t11_, v11_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
        t06_ v06_;\
        t07_ v07_;\
        t08_ v08_;\
        t09_ v09_;\
        t10_ v10_;\
        t11_ v11_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_,\
                        arg->v06_,\
                        arg->v07_,\
                        arg->v08_,\
                        arg->v09_,\
                        arg->v10_,\
                        arg->v11_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_,\
                                                                    t06_ v06_,\
                                                                    t07_ v07_,\
                                                                    t08_ v08_,\
                                                                    t09_ v09_,\
                                                                    t10_ v10_,\
                                                                    t11_ v11_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_,\
                                                 v06_,\
                                                 v07_,\
                                                 v08_,\
                                                 v09_,\
                                                 v10_,\
                                                 v11_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_)


#define MAKE_FUNC_12_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_, t06_, v06_,\
                                           t07_, v07_, t08_, v08_, t09_, v09_, t10_, v10_, t11_, v11_, t12_, v12_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_,\
                                                t12_ v12_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
        t06_ v06_;\
        t07_ v07_;\
        t08_ v08_;\
        t09_ v09_;\
        t10_ v10_;\
        t11_ v11_;\
        t12_ v12_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_,\
                        arg->v06_,\
                        arg->v07_,\
                        arg->v08_,\
                        arg->v09_,\
                        arg->v10_,\
                        arg->v11_,\
                        arg->v12_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_,\
                                                                    t06_ v06_,\
                                                                    t07_ v07_,\
                                                                    t08_ v08_,\
                                                                    t09_ v09_,\
                                                                    t10_ v10_,\
                                                                    t11_ v11_,\
                                                                    t12_ v12_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_,\
                                                 v06_,\
                                                 v07_,\
                                                 v08_,\
                                                 v09_,\
                                                 v10_,\
                                                 v11_,\
                                                 v12_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_,\
                                                t12_ v12_)


#define MAKE_FUNC_13_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_, t06_, v06_,\
                                           t07_, v07_, t08_, v08_, t09_, v09_, t10_, v10_, t11_, v11_, t12_, v12_,\
                                           t13_, v13_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_,\
                                                t12_ v12_,\
                                                t13_ v13_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
        t06_ v06_;\
        t07_ v07_;\
        t08_ v08_;\
        t09_ v09_;\
        t10_ v10_;\
        t11_ v11_;\
        t12_ v12_;\
        t13_ v13_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_,\
                        arg->v06_,\
                        arg->v07_,\
                        arg->v08_,\
                        arg->v09_,\
                        arg->v10_,\
                        arg->v11_,\
                        arg->v12_,\
                        arg->v13_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_,\
                                                                    t06_ v06_,\
                                                                    t07_ v07_,\
                                                                    t08_ v08_,\
                                                                    t09_ v09_,\
                                                                    t10_ v10_,\
                                                                    t11_ v11_,\
                                                                    t12_ v12_,\
                                                                    t13_ v13_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_,\
                                                 v06_,\
                                                 v07_,\
                                                 v08_,\
                                                 v09_,\
                                                 v10_,\
                                                 v11_,\
                                                 v12_,\
                                                 v13_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_,\
                                                t12_ v12_,\
                                                t13_ v13_)

#define MAKE_FUNC_14_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_, t06_, v06_,\
                                           t07_, v07_, t08_, v08_, t09_, v09_, t10_, v10_, t11_, v11_, t12_, v12_,\
                                           t13_, v13_, t14_, v14_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_,\
                                                t12_ v12_,\
                                                t13_ v13_,\
                                                t14_ v14_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
        t06_ v06_;\
        t07_ v07_;\
        t08_ v08_;\
        t09_ v09_;\
        t10_ v10_;\
        t11_ v11_;\
        t12_ v12_;\
        t13_ v13_;\
        t14_ v14_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_,\
                        arg->v06_,\
                        arg->v07_,\
                        arg->v08_,\
                        arg->v09_,\
                        arg->v10_,\
                        arg->v11_,\
                        arg->v12_,\
                        arg->v13_,\
                        arg->v14_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_,\
                                                                    t06_ v06_,\
                                                                    t07_ v07_,\
                                                                    t08_ v08_,\
                                                                    t09_ v09_,\
                                                                    t10_ v10_,\
                                                                    t11_ v11_,\
                                                                    t12_ v12_,\
                                                                    t13_ v13_,\
                                                                    t14_ v14_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_,\
                                                 v06_,\
                                                 v07_,\
                                                 v08_,\
                                                 v09_,\
                                                 v10_,\
                                                 v11_,\
                                                 v12_,\
                                                 v13_,\
                                                 v14_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_,\
                                                t12_ v12_,\
                                                t13_ v13_,\
                                                t14_ v14_)



#define MAKE_FUNC_15_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_, t06_, v06_,\
                                           t07_, v07_, t08_, v08_, t09_, v09_, t10_, v10_, t11_, v11_, t12_, v12_,\
                                           t13_, v13_, t14_, v14_, t15_, v15_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_,\
                                                t12_ v12_,\
                                                t13_ v13_,\
                                                t14_ v14_,\
                                                t15_ v15_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
        t06_ v06_;\
        t07_ v07_;\
        t08_ v08_;\
        t09_ v09_;\
        t10_ v10_;\
        t11_ v11_;\
        t12_ v12_;\
        t13_ v13_;\
        t14_ v14_;\
        t15_ v15_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_,\
                        arg->v06_,\
                        arg->v07_,\
                        arg->v08_,\
                        arg->v09_,\
                        arg->v10_,\
                        arg->v11_,\
                        arg->v12_,\
                        arg->v13_,\
                        arg->v14_,\
                        arg->v15_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_,\
                                                                    t06_ v06_,\
                                                                    t07_ v07_,\
                                                                    t08_ v08_,\
                                                                    t09_ v09_,\
                                                                    t10_ v10_,\
                                                                    t11_ v11_,\
                                                                    t12_ v12_,\
                                                                    t13_ v13_,\
                                                                    t14_ v14_,\
                                                                    t15_ v15_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_,\
                                                 v06_,\
                                                 v07_,\
                                                 v08_,\
                                                 v09_,\
                                                 v10_,\
                                                 v11_,\
                                                 v12_,\
                                                 v13_,\
                                                 v14_,\
                                                 v15_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_,\
                                                t12_ v12_,\
                                                t13_ v13_,\
                                                t14_ v14_,\
                                                t15_ v15_)


#define MAKE_FUNC_16_ARGS(scope__, name__, t01_, v01_, t02_, v02_, t03_, v03_, t04_, v04_, t05_, v05_, t06_, v06_,\
                                           t07_, v07_, t08_, v08_, t09_, v09_, t10_, v10_, t11_, v11_, t12_, v12_,\
                                           t13_, v13_, t14_, v14_, t15_, v15_, t16_, v16_)\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_,\
                                                t12_ v12_,\
                                                t13_ v13_,\
                                                t14_ v14_,\
                                                t15_ v15_,\
                                                t16_ v16_);\
    \
    typedef struct fox_blocks_struct_ ## name__\
    {\
        t01_ v01_;\
        t02_ v02_;\
        t03_ v03_;\
        t04_ v04_;\
        t05_ v05_;\
        t06_ v06_;\
        t07_ v07_;\
        t08_ v08_;\
        t09_ v09_;\
        t10_ v10_;\
        t11_ v11_;\
        t12_ v12_;\
        t13_ v13_;\
        t14_ v14_;\
        t15_ v15_;\
        t16_ v16_;\
    } fox_blocks_struct_ ## name__ ## _t;\
    \
    static void name__ ## _struct_func(void *a)\
    {\
        fox_blocks_struct_ ## name__ ## _t *arg = (fox_blocks_struct_ ## name__ ## _t*)a;\
        name__ ## _func(arg->v01_,\
                        arg->v02_,\
                        arg->v03_,\
                        arg->v04_,\
                        arg->v05_,\
                        arg->v06_,\
                        arg->v07_,\
                        arg->v08_,\
                        arg->v09_,\
                        arg->v10_,\
                        arg->v11_,\
                        arg->v12_,\
                        arg->v13_,\
                        arg->v14_,\
                        arg->v15_,\
                        arg->v16_);\
    }\
    \
    scope__ faux_block_t FAUX_BLOCK_FUNC make_ ## name__ ## _block (t01_ v01_,\
                                                                    t02_ v02_,\
                                                                    t03_ v03_,\
                                                                    t04_ v04_,\
                                                                    t05_ v05_,\
                                                                    t06_ v06_,\
                                                                    t07_ v07_,\
                                                                    t08_ v08_,\
                                                                    t09_ v09_,\
                                                                    t10_ v10_,\
                                                                    t11_ v11_,\
                                                                    t12_ v12_,\
                                                                    t13_ v13_,\
                                                                    t14_ v14_,\
                                                                    t15_ v15_,\
                                                                    t16_ v16_)\
    {\
        faux_block_t blk;\
        fox_blocks_struct_ ## name__ ## _t s_ = {v01_,\
                                                 v02_,\
                                                 v03_,\
                                                 v04_,\
                                                 v05_,\
                                                 v06_,\
                                                 v07_,\
                                                 v08_,\
                                                 v09_,\
                                                 v10_,\
                                                 v11_,\
                                                 v12_,\
                                                 v13_,\
                                                 v14_,\
                                                 v15_,\
                                                 v16_};\
        \
        MAKE_FUNC_HELPER(name__)\
        \
        return blk;\
    }\
    static void FAUX_BLOCK_FUNC name__ ## _func(t01_ v01_,\
                                                t02_ v02_,\
                                                t03_ v03_,\
                                                t04_ v04_,\
                                                t05_ v05_,\
                                                t06_ v06_,\
                                                t07_ v07_,\
                                                t08_ v08_,\
                                                t09_ v09_,\
                                                t10_ v10_,\
                                                t11_ v11_,\
                                                t12_ v12_,\
                                                t13_ v13_,\
                                                t14_ v14_,\
                                                t15_ v15_,\
                                                t16_ v16_)




#endif
