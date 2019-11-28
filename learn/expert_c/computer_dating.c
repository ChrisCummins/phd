#include <stdio.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
  time_t now, end = 0x7fffffff, remaining;
  int i;

  char *str = ctime(&end);

  printf("The end of time is at %s", str);

  // Disable buffering of stdout.
  setbuf(stdout, NULL);

  for (i = 0; i < 60; i++) {
    time(&now);
    remaining = (time_t)difftime(end, now);
    printf("Hurry, you only have %lu seconds left!\r", remaining);
    sleep(1);
  }
  printf("\n");

  return 0;
}
