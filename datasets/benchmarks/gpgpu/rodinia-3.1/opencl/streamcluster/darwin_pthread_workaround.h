// Based on implementation from @leo-yuriev:
// https://github.com/pmwkaa/ioarena/commit/b8854d4b164591cb62a97f67a6dc3645b26f4b39
#pragma once

#include <pthread.h>

#define PTHREAD_BARRIER_SERIAL_THREAD 1

typedef struct {
 pthread_mutex_t mutex;
 pthread_cond_t cond;
 int canary;
 int threshold;
} pthread_barrier_t;

typedef struct {} pthread_barrierattr_t;

int pthread_barrier_init(pthread_barrier_t* barrier, const pthread_barrierattr_t* attr, unsigned count);
int pthread_barrier_destroy(pthread_barrier_t* barrier);
int pthread_barrier_wait(pthread_barrier_t* barrier);

#define IOARENA_NEEDS_PTHREAD_BARRIER_IMPL
