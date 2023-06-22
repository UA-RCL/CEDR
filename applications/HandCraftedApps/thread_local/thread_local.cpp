#include <cstdio>

__attribute__((__visibility__("default"))) thread_local int __CEDR_TASK_ID__;

extern "C" void node_0(void) {
  printf("Node 0, Task ID: %d\n", __CEDR_TASK_ID__);
}
