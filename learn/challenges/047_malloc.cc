// Find the kth highest number in a binary tree.
//
// Note: assume binary search tree (i.e. complete, ordered).
#include <unistd.h>
#include <cmath>

#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"

namespace me {

struct block {
  size_t size;
  block* next;
  block* prev;

  void* mem() {
    return reinterpret_cast<void*>(reinterpret_cast<char*>(this) +
                                   sizeof(block));
  }
};

// Free list.
static block* head = nullptr;

static const long pageSize = sysconf(_SC_PAGE_SIZE);

// The minimum size of allocations.
static const size_t minAllocSize = 4 * pageSize;

// Allocates a block of size bytes of memory, returning a pointer to the
// beginning of the block.
//
// The content of the newly allocated block of memory is not initialized,
// remaining with indeterminate values.
//
// If size is zero, the return value depends on the particular library
// implementation (it may or may not be a null pointer), but the returned
// pointer shall not be dereferenced.
void* malloc(size_t size) {
  block* current = head;
  block* prev = nullptr;

  // Round the allocation to the nearest block size.
  size_t sizeWithHeader = size + sizeof(block);
  size_t numPages =
      sizeWithHeader / pageSize + (sizeWithHeader / pageSize % pageSize != 0);
  size_t allocSize = std::max(numPages * pageSize, minAllocSize);

  while (current) {
    prev = current;
    if (current->size == allocSize) {
      // Remove this block from the free list.
      if (current->prev) {
        current->prev->next = current->next;
      } else {
        head = nullptr;
      }
      if (current->next) {
        current->next->prev = current->prev;
      }
      LOG(DEBUG) << "Re-using existing block";
      break;
    }

    if (current->size > allocSize && current->size > 2 * minAllocSize) {
      // Split the block.
      size_t newBlockSize = current->size - allocSize;
      LOG(DEBUG) << "Splitting free block into two blocks of " << allocSize
                 << " and " << newBlockSize << " bytes";

      block* newBlock = reinterpret_cast<block*>(
          reinterpret_cast<char*>(current) + allocSize);

      newBlock->size = newBlockSize;
      newBlock->next = current->next;
      newBlock->prev = current;

      current->size = allocSize;
      current->next = newBlock;
      break;
    }
    current = current->next;
  }

  if (!current) {
    current = static_cast<block*>(sbrk(allocSize));
    current->size = allocSize;
    LOG(DEBUG) << "Created new block";
  }

  current->next = nullptr;
  current->prev = nullptr;

  LOG(INFO) << "malloc(" << size << ") = " << current << " (" << allocSize
            << " bytes)";

  return current->mem();
}

// A block of memory previously allocated by a call to malloc, calloc or
// realloc is deallocated, making it available again for further allocations.
//
// If ptr does not point to a block of memory allocated with the above
// functions, it causes undefined behavior.
//
// If ptr is a null pointer, the function does nothing.
//
// Notice that this function does not change the value of ptr itself, hence it
// still points to the same (now invalid) location.
void free(void* ptr) {
  block* freedBlock =
      reinterpret_cast<block*>(reinterpret_cast<char*>(ptr) - sizeof(block));

  CHECK(freedBlock->next == nullptr) << "Free block has next";
  CHECK(freedBlock->prev == nullptr) << "Free block has prev";

  LOG(INFO) << "free(" << freedBlock << ") (" << freedBlock->size << " bytes)";

  block* current = head;
  block* prev = nullptr;

  while (current) {
    prev = current;
    current = current->next;
  }

  if (prev) {
    LOG(DEBUG) << "Adding block " << freedBlock << " to free list";
    prev->next = freedBlock;
    freedBlock->prev = prev;
  } else {
    LOG(DEBUG) << "Initializing free list with " << freedBlock;
    head = freedBlock;
  }
}

namespace {

TEST(Malloc, NonNullReturn) { ASSERT_NE(malloc(10), nullptr); }

TEST(Malloc, MallocFreeReuse) {
  void* a = malloc(10);
  free(a);
  void* b = malloc(10);
  ASSERT_EQ(a, b);
  free(b);
}

TEST(Malloc, MallocNoReuse) {
  void* a = malloc(10);
  void* b = malloc(10);
  ASSERT_NE(a, b);

  free(a);
  free(b);
}

TEST(Malloc, MallocSplitFree) {
  void* a = malloc(100000);
  free(a);
  void* b = malloc(10);
  void* c = malloc(10);
  void* d = malloc(10);
  void* e = malloc(10);
  // ASSERT_EQ(a, b);
  free(b);
  free(c);
  free(d);
  free(e);
}

}  // anonymous namespace
}  // namespace me

TEST_MAIN();
