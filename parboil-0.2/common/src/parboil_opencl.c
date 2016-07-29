/*
 * (c) 2007 The Board of Trustees of the University of Illinois.
 */

#include <parboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>

#if _POSIX_VERSION >= 200112L
# include <sys/time.h>
#endif

cl_context *clContextPtr;
cl_command_queue *clCommandQueuePtr;

/* Free an array of owned strings. */
static void
free_string_array(char **string_array)
{
  char **p;

  if (!string_array) return;
  for (p = string_array; *p; p++) free(*p);
  free(string_array);
}

/* Parse a comma-delimited list of strings into an
 * array of strings. */
static char ** 
read_string_array(char *in)
{
  char **ret;
  int i;
  int count;			/* Number of items in the input */
  char *substring;		/* Current substring within 'in' */

  /* Count the number of items in the string */
  count = 1;
  for (i = 0; in[i]; i++) if (in[i] == ',') count++;

  /* Allocate storage */
  ret = (char **)malloc((count + 1) * sizeof(char *));

  /* Create copies of the strings from the list */
  substring = in;
  for (i = 0; i < count; i++) {
    char *substring_end;
    int substring_length;

    /* Find length of substring */
    for (substring_end = substring;
	 (*substring_end != ',') && (*substring_end != 0);
	 substring_end++);

    substring_length = substring_end - substring;

    /* Allocate memory and copy the substring */
    ret[i] = (char *)malloc(substring_length + 1);
    memcpy(ret[i], substring, substring_length);
    ret[i][substring_length] = 0;

    /* go to next substring */
    substring = substring_end + 1;
  }
  ret[i] = NULL;		/* Write the sentinel value */

  return ret;
}

struct argparse {
  int argc;			/* Number of arguments.  Mutable. */
  char **argv;			/* Argument values.  Immutable. */

  int argn;			/* Current argument number. */
  char **argv_get;		/* Argument value being read. */
  char **argv_put;		/* Argument value being written.
				 * argv_put <= argv_get. */
};

static void
initialize_argparse(struct argparse *ap, int argc, char **argv)
{
  ap->argc = argc;
  ap->argn = 0;
  ap->argv_get = ap->argv_put = ap->argv = argv;
}

static void
finalize_argparse(struct argparse *ap)
{
  /* Move the remaining arguments */
  for(; ap->argn < ap->argc; ap->argn++)
    *ap->argv_put++ = *ap->argv_get++;
}

/* Delete the current argument. */
static void
delete_argument(struct argparse *ap)
{
  if (ap->argn >= ap->argc) {
    //fprintf(stderr, "delete_argument\n");
  }
  ap->argc--;
  ap->argv_get++;
}

/* Go to the next argument.  Also, move the current argument to its
 * final location in argv. */
static void
next_argument(struct argparse *ap)
{
  if (ap->argn >= ap->argc) {
    //fprintf(stderr, "next_argument\n");
  }
  /* Move argument to its new location. */
  *ap->argv_put++ = *ap->argv_get++;
  ap->argn++;
}

static int
is_end_of_arguments(struct argparse *ap)
{
  return ap->argn == ap->argc;
}

static char *
get_argument(struct argparse *ap)
{
  return *ap->argv_get;
}

static char *
consume_argument(struct argparse *ap)
{
  char *ret = get_argument(ap);
  delete_argument(ap);
  return ret;
}

struct pb_Parameters *
pb_ReadParameters(int *_argc, char **argv)
{
  char *err_message;
  struct argparse ap;
  struct pb_Parameters *ret =
    (struct pb_Parameters *)malloc(sizeof(struct pb_Parameters));

  /* Initialize the parameters structure */
  ret->outFile = NULL;
  ret->inpFiles = (char **)malloc(sizeof(char *));
  ret->inpFiles[0] = NULL;

  /* Each argument */
  initialize_argparse(&ap, *_argc, argv);
  while(!is_end_of_arguments(&ap)) {
    char *arg = get_argument(&ap);

    /* Single-character flag */
    if ((arg[0] == '-') && (arg[1] != 0) && (arg[2] == 0)) {
      delete_argument(&ap);	/* This argument is consumed here */

      switch(arg[1]) {
      case 'o':			/* Output file name */
	if (is_end_of_arguments(&ap))
	  {
	    err_message = "Expecting file name after '-o'\n";
	    goto error;
	  }
	free(ret->outFile);
	ret->outFile = strdup(consume_argument(&ap));
	break;
      case 'i':			/* Input file name */
	if (is_end_of_arguments(&ap))
	  {
	    err_message = "Expecting file name after '-i'\n";
	    goto error;
	  }
	ret->inpFiles = read_string_array(consume_argument(&ap));
	break;
      case '-':			/* End of options */
	goto end_of_options;
      default:
	err_message = "Unexpected command-line parameter\n";
	goto error;
      }
    }
    else {
      /* Other parameters are ignored */
      next_argument(&ap);
    }
  } /* end for each argument */

 end_of_options:
  *_argc = ap.argc;		/* Save the modified argc value */
  finalize_argparse(&ap);

  return ret;

 error:
  fputs(err_message, stderr);
  pb_FreeParameters(ret);
  return NULL;
}

void
pb_FreeParameters(struct pb_Parameters *p)
{
  char **cpp;

  free(p->outFile);
  free_string_array(p->inpFiles);
  free(p);
}

int
pb_Parameters_CountInputs(struct pb_Parameters *p)
{
  int n;

  for (n = 0; p->inpFiles[n]; n++);
  return n;
}

/*****************************************************************************/
/* Timer routines */

static int is_async(enum pb_TimerID timer)
{
  return (timer == pb_TimerID_KERNEL) || 
             (timer == pb_TimerID_COPY_ASYNC);
}

static int is_blocking(enum pb_TimerID timer)
{
  return (timer == pb_TimerID_COPY) || (timer == pb_TimerID_NONE);
}

#define INVALID_TIMERID pb_TimerID_LAST

static int asyncs_outstanding(struct pb_TimerSet* timers)
{
  return (timers->async_markers != NULL) && 
           (timers->async_markers->timerID != INVALID_TIMERID);
}

static struct pb_async_time_marker_list * 
get_last_async(struct pb_TimerSet* timers)
{
  /* Find the last event recorded thus far */
  struct pb_async_time_marker_list * last_event = timers->async_markers;
  if(last_event != NULL && last_event->timerID != INVALID_TIMERID) {
    while(last_event->next != NULL && 
            last_event->next->timerID != INVALID_TIMERID)
      last_event = last_event->next;
    return last_event;
  } else
    return NULL;
}

static void insert_marker(struct pb_TimerSet* tset, enum pb_TimerID timer)
{
  cl_int ciErrNum = CL_SUCCESS;
  struct pb_async_time_marker_list ** new_event = &(tset->async_markers);

  while(*new_event != NULL && (*new_event)->timerID != INVALID_TIMERID) {
    new_event = &((*new_event)->next);
  }

  if(*new_event == NULL) {
    *new_event = (struct pb_async_time_marker_list *) 
      			malloc(sizeof(struct pb_async_time_marker_list));
    (*new_event)->marker = calloc(1, sizeof(cl_event));
    /*
    // I don't think this is needed at all. I believe clEnqueueMarker 'creates' the event
#if ( __OPENCL_VERSION__ >= CL_VERSION_1_1 )
fprintf(stderr, "Creating Marker [%d]\n", timer);
    *((cl_event *)((*new_event)->marker)) = clCreateUserEvent(*clContextPtr, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Creating User Event Object!\n");
    }
    ciErrNum = clSetUserEventStatus(*((cl_event *)((*new_event)->marker)), CL_QUEUED);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Setting User Event Status!\n");
    }
#endif
*/
    (*new_event)->next = NULL;
  }

  /* valid event handle now aquired: insert the event record */
  (*new_event)->label = NULL;
  (*new_event)->timerID = timer;
  ciErrNum = clEnqueueMarker(*clCommandQueuePtr, (cl_event *)(*new_event)->marker);
  if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Enqueueing Marker!\n");
  }

}

static void insert_submarker(struct pb_TimerSet* tset, char *label, enum pb_TimerID timer)
{
  cl_int ciErrNum = CL_SUCCESS;
  struct pb_async_time_marker_list ** new_event = &(tset->async_markers);

  while(*new_event != NULL && (*new_event)->timerID != INVALID_TIMERID) {
    new_event = &((*new_event)->next);
  }

  if(*new_event == NULL) {
    *new_event = (struct pb_async_time_marker_list *) 
      			malloc(sizeof(struct pb_async_time_marker_list));
    (*new_event)->marker = calloc(1, sizeof(cl_event));
    /*
#if ( __OPENCL_VERSION__ >= CL_VERSION_1_1 )
fprintf(stderr, "Creating SubMarker %s[%d]\n", label, timer);
    *((cl_event *)((*new_event)->marker)) = clCreateUserEvent(*clContextPtr, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Creating User Event Object!\n");
    }
    ciErrNum = clSetUserEventStatus(*((cl_event *)((*new_event)->marker)), CL_QUEUED);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Setting User Event Status!\n");
    }
#endif
*/
    (*new_event)->next = NULL;
  }

  /* valid event handle now aquired: insert the event record */
  (*new_event)->label = label;
  (*new_event)->timerID = timer;
  ciErrNum = clEnqueueMarker(*clCommandQueuePtr, (cl_event *)(*new_event)->marker);
  if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Enqueueing Marker!\n");
  }

}


/* Assumes that all recorded events have completed */
static pb_Timestamp record_async_times(struct pb_TimerSet* tset)
{
  struct pb_async_time_marker_list * next_interval = NULL;
  struct pb_async_time_marker_list * last_marker = get_last_async(tset);
  pb_Timestamp total_async_time = 0;
  enum pb_TimerID timer;
  
  for(next_interval = tset->async_markers; next_interval != last_marker; 
      next_interval = next_interval->next) {
    cl_ulong command_start=0, command_end=0;
    cl_int ciErrNum = CL_SUCCESS;
    
    ciErrNum = clGetEventProfilingInfo(*((cl_event *)next_interval->marker), CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &command_start, NULL);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error getting first EventProfilingInfo: %d\n", ciErrNum);
    }

    ciErrNum = clGetEventProfilingInfo(*((cl_event *)next_interval->next->marker), CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &command_end, NULL);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error getting second EventProfilingInfo: %d\n", ciErrNum);
    } 
    
    pb_Timestamp interval = (pb_Timestamp) (((double)(command_end - command_start)) / 1e3);
    tset->timers[next_interval->timerID].elapsed += interval;
    if (next_interval->label != NULL) {
      struct pb_SubTimer *subtimer = tset->sub_timer_list[next_interval->timerID]->subtimer_list;
      while (subtimer != NULL) {
        if ( strcmp(subtimer->label, next_interval->label) == 0) {
          subtimer->timer.elapsed += interval;
          break;
        }
        subtimer = subtimer->next;
      }      
    }        
    total_async_time += interval;
    next_interval->timerID = INVALID_TIMERID;
  }

  if(next_interval != NULL)
    next_interval->timerID = INVALID_TIMERID;
  
  return total_async_time;
}

static void
accumulate_time(pb_Timestamp *accum,
		pb_Timestamp start,
		pb_Timestamp end)
{
#if _POSIX_VERSION >= 200112L
  *accum += end - start;
#else
# error "Timestamps not implemented for this system"
#endif
}

#if _POSIX_VERSION >= 200112L
static pb_Timestamp get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (pb_Timestamp) (tv.tv_sec * 1000000LL + tv.tv_usec);
}
#else
# error "no supported time libraries are available on this platform"
#endif

void
pb_ResetTimer(struct pb_Timer *timer)
{
  timer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  timer->elapsed = 0;
#else
# error "pb_ResetTimer: not implemented for this system"
#endif
}

void
pb_StartTimer(struct pb_Timer *timer)
{
  if (timer->state != pb_Timer_STOPPED) {
    fputs("Ignoring attempt to start a running timer\n", stderr);
    return;
  }

  timer->state = pb_Timer_RUNNING;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    timer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
# error "pb_StartTimer: not implemented for this system"
#endif
}

void
pb_StartTimerAndSubTimer(struct pb_Timer *timer, struct pb_Timer *subtimer)
{

  unsigned int numNotStopped = 0x3; // 11
  if (timer->state != pb_Timer_STOPPED) {
    fputs("Warning: Timer was not stopped\n", stderr);
    numNotStopped &= 0x1; // Zero out 2^1
  }
  if (subtimer->state != pb_Timer_STOPPED) {
    fputs("Warning: Subtimer was not stopped\n", stderr);
    numNotStopped &= 0x2; // Zero out 2^0
  }
  if (numNotStopped == 0x0) {
    fputs("Ignoring attempt to start running timer and subtimer\n", stderr);
    return;
  }

  timer->state = pb_Timer_RUNNING;
  subtimer->state = pb_Timer_RUNNING;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    
    if (numNotStopped & 0x2) {
      timer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
    }
  
    if (numNotStopped & 0x1) {
      subtimer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
    }
  }
#else
# error "pb_StartTimer: not implemented for this system"
#endif

}

void
pb_StopTimer(struct pb_Timer *timer)
{
  pb_Timestamp fini;

  if (timer->state != pb_Timer_RUNNING) {
    fputs("Ignoring attempt to stop a stopped timer\n", stderr);
    return;
  }

  timer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    fini = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
# error "pb_StopTimer: not implemented for this system"
#endif

  accumulate_time(&timer->elapsed, timer->init, fini);
  timer->init = fini;
}

void pb_StopTimerAndSubTimer(struct pb_Timer *timer, struct pb_Timer *subtimer) {

  pb_Timestamp fini;

  unsigned int numNotRunning = 0x3; // 11
  if (timer->state != pb_Timer_RUNNING) {
    fputs("Warning: Timer was not running\n", stderr);
    numNotRunning &= 0x1; // Zero out 2^1
  }
  if (subtimer->state != pb_Timer_RUNNING) {
    fputs("Warning: Subtimer was not running\n", stderr);
    numNotRunning &= 0x2; // Zero out 2^0
  }
  if (numNotRunning == 0x0) {
    fputs("Ignoring attempt to stop stopped timer and subtimer\n", stderr);
    return;
  }


  timer->state = pb_Timer_STOPPED;
  subtimer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    fini = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
# error "pb_StopTimer: not implemented for this system"
#endif

  if (numNotRunning & 0x2) {
    accumulate_time(&timer->elapsed, timer->init, fini);
    timer->init = fini;
  }
  
  if (numNotRunning & 0x1) {
    accumulate_time(&subtimer->elapsed, subtimer->init, fini);
    subtimer->init = fini;
  }

}

/* Get the elapsed time in seconds. */
double
pb_GetElapsedTime(struct pb_Timer *timer)
{
  double ret;

  if (timer->state != pb_Timer_STOPPED) {
    fputs("Elapsed time from a running timer is inaccurate\n", stderr);
  }

#if _POSIX_VERSION >= 200112L
  ret = timer->elapsed / 1e6;
#else
# error "pb_GetElapsedTime: not implemented for this system"
#endif
  return ret;
}

void
pb_InitializeTimerSet(struct pb_TimerSet *timers)
{
  int n;

  timers->wall_begin = get_time();
  timers->current = pb_TimerID_NONE;

  timers->async_markers = NULL;
  
  for (n = 0; n < pb_TimerID_LAST; n++) {
    pb_ResetTimer(&timers->timers[n]);
    timers->sub_timer_list[n] = NULL;
  }
}

void pb_SetOpenCL(void *p_clContextPtr, void *p_clCommandQueuePtr) {
  clContextPtr = ((cl_context *)p_clContextPtr);
  clCommandQueuePtr = ((cl_command_queue *)p_clCommandQueuePtr);
}

void
pb_AddSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID pb_Category) {    
  
  struct pb_SubTimer *subtimer = (struct pb_SubTimer *) malloc
    (sizeof(struct pb_SubTimer));
     
  int len = strlen(label);
    
  subtimer->label = (char *) malloc (sizeof(char)*(len+1));
  sprintf(subtimer->label, "%s\0", label);
  
  pb_ResetTimer(&subtimer->timer);
  subtimer->next = NULL;
  
  struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[pb_Category];
  if (subtimerlist == NULL) {
    subtimerlist = (struct pb_SubTimerList *) calloc
      (1, sizeof(struct pb_SubTimerList));
    subtimerlist->subtimer_list = subtimer;
    timers->sub_timer_list[pb_Category] = subtimerlist;
  } else {
    // Append to list
    struct pb_SubTimer *element = subtimerlist->subtimer_list;
    while (element->next != NULL) {
      element = element->next;
    }
    element->next = subtimer;
  }
  
}

void
pb_SwitchToTimer(struct pb_TimerSet *timers, enum pb_TimerID timer)
{
  /* Stop the currently running timer */
  if (timers->current != pb_TimerID_NONE) {
    struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[timers->current];
    struct pb_SubTimer *currSubTimer = (subtimerlist != NULL) ? subtimerlist->current : NULL;
  
    if (!is_async(timers->current) ) {
      if (timers->current != timer) {
        if (currSubTimer != NULL) {
          pb_StopTimerAndSubTimer(&timers->timers[timers->current], &currSubTimer->timer);
        } else {
          pb_StopTimer(&timers->timers[timers->current]);
        }
      } else {
        if (currSubTimer != NULL) {
          pb_StopTimer(&currSubTimer->timer);
        }
      }
    } else {
      insert_marker(timers, timer);
      if (!is_async(timer)) { // if switching to async too, keep driver going
        pb_StopTimer(&timers->timers[pb_TimerID_DRIVER]);
      }
    }
  }
  
  pb_Timestamp currentTime = get_time();

  /* The only cases we check for asynchronous task completion is 
   * when an overlapping CPU operation completes, or the next 
   * segment blocks on completion of previous async operations */
  if( asyncs_outstanding(timers) && 
      (!is_async(timers->current) || is_blocking(timer) ) ) {

    struct pb_async_time_marker_list * last_event = get_last_async(timers);
    /* CL_COMPLETE if completed */
    
    cl_int ciErrNum = CL_SUCCESS;
    cl_int async_done = CL_COMPLETE;
    
    ciErrNum = clGetEventInfo(*((cl_event *)last_event->marker), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &async_done, NULL);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Querying EventInfo!\n");
    }
    

    if(is_blocking(timer)) {
      /* Async operations completed after previous CPU operations: 
       * overlapped time is the total CPU time since this set of async 
       * operations were first issued */
       
      // timer to switch to is COPY or NONE 
      if(async_done != CL_COMPLETE) {
        accumulate_time(&(timers->timers[pb_TimerID_OVERLAP].elapsed), 
	                  timers->async_begin,currentTime);
      }	                  
	                  
      /* Wait on async operation completion */
      ciErrNum = clWaitForEvents(1, (cl_event *)last_event->marker);
      if (ciErrNum != CL_SUCCESS) {
        fprintf(stderr, "Error Waiting for Events!\n");
      }
      
      pb_Timestamp total_async_time = record_async_times(timers);

      /* Async operations completed before previous CPU operations: 
       * overlapped time is the total async time */
      if(async_done == CL_COMPLETE) {
        //fprintf(stderr, "Async_done: total_async_type = %lld\n", total_async_time);
        timers->timers[pb_TimerID_OVERLAP].elapsed += total_async_time;
      }

    } else 
    /* implies (!is_async(timers->current) && asyncs_outstanding(timers)) */
    // i.e. Current Not Async (not KERNEL/COPY_ASYNC) but there are outstanding
    // so something is deeper in stack
    if(async_done == CL_COMPLETE ) {
      /* Async operations completed before previous CPU operations: 
       * overlapped time is the total async time */
      timers->timers[pb_TimerID_OVERLAP].elapsed += record_async_times(timers);
    }   
  }

  /* Start the new timer */
  if (timer != pb_TimerID_NONE) {
    if(!is_async(timer)) {
      pb_StartTimer(&timers->timers[timer]);
    } else {
      // toSwitchTo Is Async (KERNEL/COPY_ASYNC)
      if (!asyncs_outstanding(timers)) {
        /* No asyncs outstanding, insert a fresh async marker */
      
        insert_marker(timers, timer);
        timers->async_begin = currentTime;
      } else if(!is_async(timers->current)) {
        /* Previous asyncs still in flight, but a previous SwitchTo
         * already marked the end of the most recent async operation, 
         * so we can rename that marker as the beginning of this async 
         * operation */
         
        struct pb_async_time_marker_list * last_event = get_last_async(timers);
        last_event->label = NULL;
        last_event->timerID = timer;
      }
      if (!is_async(timers->current)) {
        pb_StartTimer(&timers->timers[pb_TimerID_DRIVER]);
      }
    }
  }
  timers->current = timer;

}

void
pb_SwitchToSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID category) 
{
  struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[timers->current];
  struct pb_SubTimer *curr = (subtimerlist != NULL) ? subtimerlist->current : NULL;
  
  if (timers->current != pb_TimerID_NONE) {
    if (!is_async(timers->current) ) {
      if (timers->current != category) {
        if (curr != NULL) {
          pb_StopTimerAndSubTimer(&timers->timers[timers->current], &curr->timer);
        } else {
          pb_StopTimer(&timers->timers[timers->current]);
        }
      } else {
        if (curr != NULL) {
          pb_StopTimer(&curr->timer);
        }
      }
    } else {
      insert_submarker(timers, label, category);
      if (!is_async(category)) { // if switching to async too, keep driver going
        pb_StopTimer(&timers->timers[pb_TimerID_DRIVER]);
      }
    }
  }

  pb_Timestamp currentTime = get_time();

  /* The only cases we check for asynchronous task completion is 
   * when an overlapping CPU operation completes, or the next 
   * segment blocks on completion of previous async operations */
  if( asyncs_outstanding(timers) && 
      (!is_async(timers->current) || is_blocking(category) ) ) {

    struct pb_async_time_marker_list * last_event = get_last_async(timers);
    /* CL_COMPLETE if completed */

    cl_int ciErrNum = CL_SUCCESS;
    cl_int async_done = CL_COMPLETE;
    
    ciErrNum = clGetEventInfo(*((cl_event *)last_event->marker), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &async_done, NULL);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Querying EventInfo!\n");
    }

    if(is_blocking(category)) {
      /* Async operations completed after previous CPU operations: 
       * overlapped time is the total CPU time since this set of async 
       * operations were first issued */
       
      // timer to switch to is COPY or NONE 
      // if it hasn't already finished, then just take now and use that as the elapsed time in OVERLAP
      // anything happening after now isn't OVERLAP because everything is being stopped to wait for synchronization
      // it seems that the extra sync wall time isn't being recorded anywhere
      if(async_done != CL_COMPLETE) 
        accumulate_time(&(timers->timers[pb_TimerID_OVERLAP].elapsed), 
	                  timers->async_begin,currentTime);

      /* Wait on async operation completion */
      ciErrNum = clWaitForEvents(1, (cl_event *)last_event->marker);
      if (ciErrNum != CL_SUCCESS) {
        fprintf(stderr, "Error Waiting for Events!\n");
      }
      pb_Timestamp total_async_time = record_async_times(timers);

      /* Async operations completed before previous CPU operations: 
       * overlapped time is the total async time */
       // If it did finish, then accumulate all the async time that did happen into OVERLAP
       // the immediately preceding EventSynchronize theoretically didn't have any effect since it was already completed.
      if(async_done == CL_COMPLETE /*cudaSuccess*/)
        timers->timers[pb_TimerID_OVERLAP].elapsed += total_async_time;

    } else 
    /* implies (!is_async(timers->current) && asyncs_outstanding(timers)) */
    // i.e. Current Not Async (not KERNEL/COPY_ASYNC) but there are outstanding
    // so something is deeper in stack
    if(async_done == CL_COMPLETE /*cudaSuccess*/) {
      /* Async operations completed before previous CPU operations: 
       * overlapped time is the total async time */
      timers->timers[pb_TimerID_OVERLAP].elapsed += record_async_times(timers);
    }   
    // else, this isn't blocking, so just check the next time around
  }
  
  subtimerlist = timers->sub_timer_list[category];
  struct pb_SubTimer *subtimer = NULL;
  
  if (label != NULL) {
    subtimer = subtimerlist->subtimer_list;
    while (subtimer != NULL) {
      if (strcmp(subtimer->label, label) == 0) {
        break;
      } else {
        subtimer = subtimer->next;
      }
    }
  }

  /* Start the new timer */
  if (category != pb_TimerID_NONE) {
    if(!is_async(category)) {  
      if (subtimerlist != NULL) {
        subtimerlist->current = subtimer;
      }
    
      if (category != timers->current && subtimer != NULL) {
        pb_StartTimerAndSubTimer(&timers->timers[category], &subtimer->timer);
      } else if (subtimer != NULL) {
        pb_StartTimer(&subtimer->timer);
      } else {
        pb_StartTimer(&timers->timers[category]);
      }
    } else {
      if (subtimerlist != NULL) {
        subtimerlist->current = subtimer;
      }
    
      // toSwitchTo Is Async (KERNEL/COPY_ASYNC)
      if (!asyncs_outstanding(timers)) {
        /* No asyncs outstanding, insert a fresh async marker */
        insert_submarker(timers, label, category);
        timers->async_begin = currentTime;
      } else if(!is_async(timers->current)) {
        /* Previous asyncs still in flight, but a previous SwitchTo
         * already marked the end of the most recent async operation, 
         * so we can rename that marker as the beginning of this async 
         * operation */
                  
        struct pb_async_time_marker_list * last_event = get_last_async(timers);
        last_event->timerID = category;
        last_event->label = label;
      } // else, marker for switchToThis was already inserted
      
      //toSwitchto is already asynchronous, but if current/prev state is async too, then DRIVER is already running
      if (!is_async(timers->current)) {
        pb_StartTimer(&timers->timers[pb_TimerID_DRIVER]);
      }
    }
  }
  
  timers->current = category;
}

void
pb_PrintTimerSet(struct pb_TimerSet *timers)
{
  pb_Timestamp wall_end = get_time();

  struct pb_Timer *t = timers->timers;
  struct pb_SubTimer* sub = NULL;
  
  int maxSubLength;
    
  const char *categories[] = {
    "IO", "Kernel", "Copy", "Driver", "Copy Async", "Compute"
  };
  
  const int maxCategoryLength = 10;
  
  int i;
  for(i = 1; i < pb_TimerID_LAST-1; ++i) { // exclude NONE and OVRELAP from this format
    if(pb_GetElapsedTime(&t[i]) != 0) {
    
      // Print Category Timer
      printf("%-*s: %f\n", maxCategoryLength, categories[i-1], pb_GetElapsedTime(&t[i]));
      
      if (timers->sub_timer_list[i] != NULL) {
        sub = timers->sub_timer_list[i]->subtimer_list;
        maxSubLength = 0;
        while (sub != NULL) {
          // Find longest SubTimer label
          if (strlen(sub->label) > maxSubLength) {
            maxSubLength = strlen(sub->label);
          }
          sub = sub->next;
        }
        
        // Fit to Categories
        if (maxSubLength <= maxCategoryLength) {
         maxSubLength = maxCategoryLength;
        }
        
        sub = timers->sub_timer_list[i]->subtimer_list;
        
        // Print SubTimers
        while (sub != NULL) {
          printf(" -%-*s: %f\n", maxSubLength, sub->label, pb_GetElapsedTime(&sub->timer));
          sub = sub->next;
        }
      }
    }
  }
  
  if(pb_GetElapsedTime(&t[pb_TimerID_OVERLAP]) != 0)
    printf("CPU/Kernel Overlap: %f\n", pb_GetElapsedTime(&t[pb_TimerID_OVERLAP]));
        
  float walltime = (wall_end - timers->wall_begin)/ 1e6;
  printf("Timer Wall Time: %f\n", walltime);
  
}

void pb_DestroyTimerSet(struct pb_TimerSet * timers)
{
  /* clean up all of the async event markers */
  struct pb_async_time_marker_list* event = timers->async_markers;
  while(event != NULL) {

    cl_int ciErrNum = CL_SUCCESS;
    ciErrNum = clWaitForEvents(1, (cl_event *)(event)->marker);
    if (ciErrNum != CL_SUCCESS) {
      //fprintf(stderr, "Error Waiting for Events!\n");
    }
    
    ciErrNum = clReleaseEvent( *((cl_event *)(event)->marker) );
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Release Events!\n");
    }
    
    free((event)->marker);
    struct pb_async_time_marker_list* next = ((event)->next);

    free(event);

    // (*event) = NULL;
    event = next;
  }

  int i = 0;
  for(i = 0; i < pb_TimerID_LAST; ++i) {    
    if (timers->sub_timer_list[i] != NULL) {
      struct pb_SubTimer *subtimer = timers->sub_timer_list[i]->subtimer_list;
      struct pb_SubTimer *prev = NULL;
      while (subtimer != NULL) {
        free(subtimer->label);
        prev = subtimer;
        subtimer = subtimer->next;
        free(prev);
      }
      free(timers->sub_timer_list[i]);
    }
  }
}


