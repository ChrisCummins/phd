#include <iostream>

#include <unistd.h>
#include <sys/wait.h>

int main() {
  pid_t pid = fork();

  switch (pid) {
    case -1:
      std::cerr << "fork() failed" << std::endl;
      exit(1);
    case 0:
      // Run program:
      execl("./functional", NULL);
      std::cerr << "execl() failed" << std::endl;
      exit(1);
    default:
      int status{0};

      std::cout << "Process created with pid " << pid << std::endl;

      // Wait for process to finish:
      while (!WIFEXITED(status))
        waitpid(pid, &status, 0);

      std::cout << "Process exited with " << WEXITSTATUS(status) << std::endl;

      return 0;
  }
}
