/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <alloca.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <pthread.h>

#include "FauxBlock.h"

typedef void *(*pthrd_start_func) (void *);    // Used by pthreads as the thread func signature

//=====================================================
//=====================================================
//=====================================================
//=====================================================

#include <stdlib.h>

typedef void (*exec_block_func) (faux_block_t *);

typedef struct async_runner
{
             pthread_t        t;
             pthread_mutex_t  m;
             pthread_cond_t   go;
    volatile unsigned int     ready;

             pthread_mutex_t  m2;
             pthread_cond_t   done;

    volatile exec_block_func   f;
    volatile void *           d;
} async_runner_t;



#define MAX_CACHED_RUNNERS 12 
static async_runner_t* runner_stack[MAX_CACHED_RUNNERS];
static int next = -1;


static void dispose_runner(async_runner_t *r)
{
    pthread_mutex_destroy (&(r->m));
    pthread_mutex_destroy (&(r->m2));
    pthread_cond_destroy  (&(r->go));
    pthread_cond_destroy  (&(r->done));
    pthread_detach((r->t));

    free (r);
}

/**
 * This is the thread code that loops forever (more or less) in the
 * runner threads.
 */
static void *thread_pool_thread(void *data)
{
    async_runner_t* r = (async_runner_t*)data;

    exec_block_func  f;
    void*            d;

    pthread_mutex_lock (&(r->m));                   // Lock
    do
    {
        // Only wait if we don't have ready signaled.  If we don't
        // have the protective 'if', we can lose signals and hang.
        if (r->ready == 0)
            pthread_cond_wait(&(r->go), &(r->m));       // Wait ... mutex unlocked inside, locked before return ...

        f = r->f;
        if (f == 0)
            goto done;

        d = (void*)(r->d);

        (*f)((faux_block_t*)d);

        r->f = 0;
        r->d = 0;
        r->ready = 0;
        // Signal that we are done.  The join() method cares ...
        pthread_mutex_lock(&(r->m2));
        pthread_cond_signal(&(r->done));
        pthread_mutex_unlock(&(r->m2));
    } while (1);

done:
    dispose_runner(r);

    pthread_exit(0);
    return 0;
}

static pthread_mutex_t runner_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * Block until the runner thread has finished handling its current task.
 */
static void join (async_runner_t *runner)
{
    // Wait (if necessary) for the runner thread to tell us it is finished
    // running its current task.

    pthread_mutex_lock(&(runner->m2));
    if (runner->f)
        pthread_cond_wait(&(runner->done), &(runner->m2));
    pthread_mutex_unlock(&(runner->m2));

    // Get back the mutex from the worker...
    pthread_mutex_lock (&(runner->m));

    pthread_mutex_lock(&runner_pool_mutex);
    // Return the runnable to the stack of available runnables.
    if (next < (MAX_CACHED_RUNNERS - 1))
    {
        ++next;
        runner_stack[next] = runner;
        pthread_mutex_unlock(&runner_pool_mutex);
    }
    else
    {
        pthread_mutex_unlock(&runner_pool_mutex);

        // "Run" this again, with func as null (which is how the thread left things).
        // With a null func, the thread will exit, which is what we want since we
        // can't access it after we fail to put it back on the stack.

        pthread_cond_signal(&(runner->go));
        pthread_mutex_unlock (&(runner->m));          // Allow the worker thread to run
    }
}





// async_runner_t* run_async (pthrd_start_func func, void *data)
static async_runner_t* run_async (exec_block_func func, void *data)
{
    async_runner_t *r = 0;

    pthread_mutex_lock(&runner_pool_mutex);
    if (next >= 0)
    {
        r = runner_stack[next];
        --next;
    }
    pthread_mutex_unlock(&runner_pool_mutex);

    if (r == 0)
    {
        r = (async_runner_t*)malloc (sizeof(async_runner_t));

        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        pthread_mutex_init (&(r->m), 0);
        pthread_cond_init  (&(r->go), 0);

        pthread_mutex_init (&(r->m2), 0);
        pthread_cond_init  (&(r->done), 0);
        r->ready = 1;
        pthread_mutex_lock (&(r->m));

       pthread_create(&(r->t), &attr, thread_pool_thread, r);
    }

    r->f = func;
    r->d = data;

    // Already locked ... either from the block above, or from the join() method
    pthread_cond_signal(&(r->go));
    pthread_mutex_unlock (&(r->m));          // Allow the worker thread to run

    return r;
}
//======================
//======================
//======================
//======================


/**
 *
 */
void clear_runners()
{
    while (next >= 0)
    {
        async_runner_t *r = runner_stack[next];
        pthread_cond_signal(&(r->go));
        pthread_mutex_unlock (&(r->m));          // Allow the worker thread to run
        --next;
        // dispose_runner(r);
    }
}


/**
 * Execute 'block' on this thread (so don't spin a worker thread)..
 */
void exec_faux_block (faux_block_t *block)
{
    block->func((void*)(block->args));
}

//## TODO: Handle call when block still assigned to other thread


/**
 * Execute 'len' of the passed in 'block' FauxBlocks, each on its own thread.
 */
int exec_faux_block_deferred (faux_block_t block[], size_t len)
{
// printf("exec_faux_block_deferred(%p, %d)\n", block, len);

    size_t i;

    for (i = 0; i < len; ++i)
    {
        async_runner_t *r = run_async (exec_faux_block, block + i);
        block[i].tid = r; // t;       // join will want this ...
    }

    //## TODO Handle errors ...
    return 0;
}



void *thread_pool_thread(void *data);
/**
 * Wait until 'len' of the passed in 'block' FauxBlocks are done executing..
 */
void faux_block_exec_join(faux_block_t block[], size_t len)
{
    //## TODO: Handle join w/o tid set

    size_t i;

    for (i = 0; i < len; ++i)
    {
        join ((async_runner_t*) block[i].tid);
        block[i].tid = 0;                  // Zero out the tid ... we aren't running in a thread anymore
    }
}


void faux_block_run (faux_block_t block[], size_t len)
{
    exec_faux_block_deferred (block, len);
    faux_block_exec_join     (block, len);
}



void not_enough_arg_space(void *arg)
{
    fprintf(stderr, "Arguments to function too large.\n");    //##TODO: It would be nice to pass the function name ...
    exit(-50);\
}
