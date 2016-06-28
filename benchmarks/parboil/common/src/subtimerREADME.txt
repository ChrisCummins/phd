
Old             |   New
----------------------------------
initTimers()    |   initTimers()
                |   addSubTimer(Kernel1, GPU)
                |   createSubTimer(Kernel2, GPU)
               ...
switchTo(GPU)   |   switchTo(Kernel1, GPU)
kernel1<<<>>>   |   kernel1<<<>>>
                    switchTo(Kernel2, GPU)  
kernel2<<<>>>   |   kernel2<<<>>>
switchTo(COPY)  |   switchTo(COPY)

-----------------------------------------

pb_InitializeTimerSet(&timers);

char *prescans = "PreScanKernel";
  
pb_AddSubTimer(&timers, prescans, pb_TimerID_GPU);
...
pb_SwitchToTimer(&timers, pb_TimerID_IO);
...
pb_SwitchToSubTimer(&timers, prescans , pb_TimerID_GPU); // this would also switch to ID_GPU if the timer is not already running
...
pb_SwitchToSubTimer(&timers, NULL , pb_TimerID_GPU); // stops subtimer if already on GPU, but GPU still runs
...
pb_SwitchToTimer(&timers, pb_TimerID_IO); // this would stop the running timer and subtimer (if any)


switchToSubTimer
 -- find the subTimer
 -- if category timer was not already running,
   * current timer is stopped
    / Async handling?
   * category timer is started
 -- if category timer was already running,
   * locate currently running subtimer
   * stop it
 -- starts this subTimer
 
 // only one subtimer of a category run at a time
 

