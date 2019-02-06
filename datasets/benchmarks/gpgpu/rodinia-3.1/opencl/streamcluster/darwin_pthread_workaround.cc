#include "./darwin_pthread_workaround.h"

#ifdef DARWIN_PTHREAD_WORKAROUND_NEEDED

#include <errno.h>
#include <assert.h>

int pthread_barrier_init(pthread_barrier_t* barrier, const pthread_barrierattr_t* attr, unsigned count) {
 if (count == 0) {
   errno = EINVAL;
   return -1;
 }
 if (attr != NULL) {
   errno = ENOSYS;
   return -1;
 }

 if (pthread_mutex_init(&barrier->mutex, NULL) < 0) {
   return -1;
 }
 if (pthread_cond_init(&barrier->cond, NULL) < 0) {
   pthread_mutex_destroy(&barrier->mutex);
   return -1;
 }

 barrier->threshold = count;
 barrier->canary = 0;
 return 0;
}

int pthread_barrier_destroy(pthread_barrier_t* barrier)
{
 barrier->threshold = -1;
 pthread_cond_destroy(&barrier->cond);
 return pthread_mutex_destroy(&barrier->mutex);
}

int pthread_barrier_wait(pthread_barrier_t* barrier)
{
 int rc = pthread_mutex_lock(&barrier->mutex);
 if (rc == 0) {
   assert(barrier->threshold > 0);
   assert(barrier->canary >= 0 && barrier->canary < barrier->threshold);

   if (++barrier->canary == barrier->threshold) {
     barrier->canary = 0;
     pthread_cond_broadcast(&barrier->cond);
     rc = PTHREAD_BARRIER_SERIAL_THREAD;
   } else {
     pthread_cond_wait(&barrier->cond, &barrier->mutex);
   }

   pthread_mutex_unlock(&barrier->mutex);
 }
 return rc;
}

#endif // DARWIN_PTHREAD_WORKAROUND_NEEDED
