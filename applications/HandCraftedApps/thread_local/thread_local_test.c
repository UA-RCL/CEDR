#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <pthread.h>

struct arg_struct {
  void* dlhandle;
};

void* threadfunc(void *arg) {
  struct arg_struct *arg_struct = arg;
  void* dlhandle = arg_struct->dlhandle;
  int* var;
  char* error;

  var = (int*) dlsym(dlhandle, "__CEDR_TASK_ID__"); 

  if ((error = dlerror()) != NULL) {
    fprintf(stderr, "Couldn't get symbol: %s\n", error);
    exit(EXIT_FAILURE);
  }

  return 0;
}

int main(void) {
  void *dlhandle;
  int* var;
  pthread_t thread;
  int pthread_err;
  struct arg_struct args;
  
  dlhandle = dlopen("../../target/x86/thread_local.so", RTLD_LAZY);
  if (!dlhandle) {
    fprintf(stderr, "Couldn't open so: %s\n", dlerror());
    exit(EXIT_FAILURE);
  } 

  args.dlhandle = dlhandle;
  
  dlerror();

  pthread_err = pthread_create(&thread, NULL, threadfunc, &args);

  if (!pthread_err) {
    pthread_join(thread, NULL);
  }

  printf("I'm able to access the __CEDR_TASK_ID__ variable from inside the shared object\n");
  return 0;
}
