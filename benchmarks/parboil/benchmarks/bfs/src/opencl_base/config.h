#define MAX_THREADS_PER_BLOCK 256
#define LOCAL_MEM_SIZE 1600 //This needs to be adjusted for certain graphs with high degrees
#define INF 2147483647//2^31-1
#define UP_LIMIT 16677216//2^24
#define WHITE 16677217
#define GRAY 16677218
#define GRAY0 16677219
#define GRAY1 16677220
#define BLACK 16677221

struct Node {
  int x;
  int y;
};
struct Edge {
  int x;
  int y;
};
