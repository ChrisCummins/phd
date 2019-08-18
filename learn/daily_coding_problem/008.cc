// This problem was asked by Apple.
//
// Implement a job scheduler which takes in a function f and an integer n, and
// calls f after n milliseconds.
#include <unistd.h>
#include <ctime>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

const unsigned int MICROSECONDS_IN_MILLISECOND = 1000;

template <typename Function, typename... Args>
void RunAfterDelay(const int milliseconds, Function&& f, Args&&... args) {
  std::cout << "Sleeping for " << milliseconds << " milliseconds ...\n";
  usleep(milliseconds * MICROSECONDS_IN_MILLISECOND);
  std::cout << "... slept\n";
  f(std::forward<Args>(args)...);
}

class Scheduler {
 public:
  template <typename Function, typename... Args>
  void Schedule(const int milliseconds, Function&& f, Args&&... args) {
    // Lambda captures by value so that memory is copied.
    threads_.push_back(
        std::thread([=] { RunAfterDelay(milliseconds, f, args...); }));
  }

  void Join() {
    std::cout << "Join ...\n";
    for (auto& thread : threads_) {
      thread.join();
    }
    std::cout << "done\n";
  }

 private:
  std::vector<std::thread> threads_;
};

int main(int argc, char** argv) {
  Scheduler scheduler;

  scheduler.Schedule(2000, [](int x, int y) { return x + y; }, 5, 5);
  std::cout << "Scheduled\n";

  scheduler.Join();

  return 0;
}
