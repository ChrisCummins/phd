// Challenge: implement std::shared_ptr.
#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"

#include <atomic>
#include <utility>

template <typename T>
struct shared_ptr {
 private:
  T* const ptr_;
  // Atomic refcounter for thread-safe refcount increment / decrement.
  std::atomic<size_t>* const rc_;

 public:
  T* const get() const {
    LOG(INFO) << "get()";
    return ptr_;
  }

  explicit shared_ptr(T* const ptr)
      : ptr_(ptr), rc_(new std::atomic<size_t>(1)) {
    LOG(INFO) << "explicit shared_ptr()";
  }

  shared_ptr(shared_ptr& p) : ptr_(p.ptr_), rc_(p.rc_) {
    ++(*rc_);
    LOG(INFO) << "copy constructor, refcount " << *rc_;
  }

  shared_ptr(shared_ptr&& p) : ptr_(p.ptr_), rc_(p.rc_) {
    ++(*rc_);
    LOG(INFO) << "move constructor, refcount " << *rc_;
  }

  ~shared_ptr() {
    --(*rc_);
    // Although this access to refcount permits races, the only situation where
    // refcount is zero is when this currently executing destructor holds the
    // last remaining reference, so there's no need for a lock here.
    if (*rc_) {
      LOG(INFO) << "~shared_ptr(), refcount " << *rc_;
    } else {
      LOG(INFO) << "~shared_ptr(), refcount 0, freeing";
      delete ptr_;
      delete rc_;
    }
  }
};

template <typename T, typename... Args>
shared_ptr<T> make_shared(Args&&... args) {
  return shared_ptr<T>(new T(std::forward<Args>(args)...));
}

TEST(shared_ptr, ExplicitConstructor) {
  shared_ptr<int> p(new int(5));
  EXPECT_EQ(*p.get(), 5);
}

TEST(shared_ptr, MutatePointerData) {
  shared_ptr<int> p(new int(5));
  *p.get() = 10;
  EXPECT_EQ(*p.get(), 10);
}

TEST(shared_ptr, MoveConstructor) {
  shared_ptr<int> p(new int(5));
  shared_ptr<int> u = std::move(p);
  EXPECT_EQ(*u.get(), 5);
}

TEST(shared_ptr, CopyConstructor) {
  shared_ptr<int> p(new int(5));
  shared_ptr<int> u = p;
  EXPECT_EQ(*u.get(), 5);
}

TEST(shared_ptr, make_shared) {
  auto p = make_shared<int>(5);
  EXPECT_EQ(*p.get(), 5);
}

TEST_MAIN();
