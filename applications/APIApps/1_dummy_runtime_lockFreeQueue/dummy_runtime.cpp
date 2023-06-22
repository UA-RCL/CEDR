#include <vector>
#include <string>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>

#include "dash.h"

#include <xenium/nikolaev_queue.hpp>
#include <xenium/reclamation/generic_epoch_based.hpp>

#define MAX_ARGS 15

//------------------------ Start Kernel Implementations -----------------------//

void do_fft_cpu(double** input_ptr, double** output_ptr, size_t *size_ptr, bool *forwardTrans_ptr) {
  double* input = *input_ptr;
  double* output = *output_ptr;
  size_t fft_size = *size_ptr;
  bool forwardTrans = *forwardTrans_ptr;

  DASH_FFT(input, output, fft_size, forwardTrans);
}

void do_mmult_cpu(double **A, double **Ai, double **B, double **Bi, double **C, double **Ci, size_t *A_Row, size_t *A_Col, size_t *B_Col) {
  double *A_re = *A;
  double *A_im = *Ai;
  double *B_re = *B;
  double *B_im = *Bi;
  double *C_re = *C;
  double *C_im = *Ci;
  const size_t A_ROWS = *A_Row;
  const size_t A_COLS = *A_Col;
  const size_t B_COLS = *B_Col;

  DASH_GEMM(A_re, A_im, B_re, B_im, C_re, C_im, A_ROWS, A_COLS, B_COLS);
}

//------------------------- End Kernel Implementations ------------------------//

struct task_node_t {
  std::string name;
  std::vector<void*> args;
  void* run_function;
  pthread_barrier_t* completion_barrier;
};
typedef struct task_node_t task_node;

typedef xenium::nikolaev_queue<
    task_node*,
    xenium::policy::reclaimer<xenium::reclamation::epoch_based<>>,
    xenium::policy::entries_per_node<2048>    // INFO: Number of concurrent threads allowed to operate on queue
    > lockFree_queue;
lockFree_queue task_list;

extern "C" void enqueue_kernel(const char* kernel_name, ...) {
  std::string kernel_str(kernel_name);

  va_list args;
  va_start(args, kernel_name);

  if (kernel_str == "DASH_FFT") {
    printf("[nk] I am inside the runtime's codebase, unpacking my args to enqueue a new FFT task\n");

    double** input = va_arg(args, double**);
    double** output = va_arg(args, double**);
    size_t* size = va_arg(args, size_t*);
    bool* forwardTrans = va_arg(args, bool*);
    // Last arg: needs to be the synchronization barrier
    pthread_barrier_t* barrier = va_arg(args, pthread_barrier_t*);
    va_end(args);

    // Create some sort of task node to represent this task
    task_node* new_node = (task_node*) calloc(1, sizeof(task_node));
    new_node->name = "FFT";
    new_node->args.push_back(input);
    new_node->args.push_back(output);
    new_node->args.push_back(size);
    new_node->args.push_back(forwardTrans);
    new_node->run_function = (void*) do_fft_cpu;
    new_node->completion_barrier = barrier; 
    
    printf("[nk] I have finished initializing my FFT node, pushing it onto the task list\n");

    // Push this node onto the ready queue
    task_list.push(new_node);
    printf("[nk] I have pushed a new task onto the work queue, time to go sleep until it gets scheduled and completed\n");
  } else if (kernel_str == "DASH_GEMM") {
    // Unpack args and enqueue an mmult task
    double** A_re = va_arg(args, double**);
    double** A_im = va_arg(args, double**);
    double** B_re = va_arg(args, double**);
    double** B_im = va_arg(args, double**);
    double** C_re = va_arg(args, double**);
    double** C_im = va_arg(args, double**);
    size_t* A_Rows = va_arg(args, size_t*);
    size_t* A_Cols = va_arg(args, size_t*);
    size_t* B_Cols = va_arg(args, size_t*);
    // Last arg: needs to be the synchronization barrier
    pthread_barrier_t* barrier = va_arg(args, pthread_barrier_t*);
    va_end(args);

    // Create some sort of task node to represent this task
    task_node* new_node = (task_node*) calloc(1, sizeof(task_node));
    new_node->name = "GEMM";
    new_node->args.push_back(A_re);
    new_node->args.push_back(A_im);
    new_node->args.push_back(B_re);
    new_node->args.push_back(B_im);
    new_node->args.push_back(C_re);
    new_node->args.push_back(C_im);
    new_node->args.push_back(A_Rows);
    new_node->args.push_back(A_Cols);
    new_node->args.push_back(B_Cols);
    new_node->run_function = (void*) do_mmult_cpu;
    new_node->completion_barrier = barrier;

    printf("[nk] I have finished initializing my GEMM node, pushing it onto the task list\n");

    // Push this node onto the ready queue
    task_list.push(new_node);
    printf("[nk] I have pushed a new task onto the work queue, time to go sleep until it gets scheduled and completed\n");
  } else if (kernel_str == "DASH_FIR") {
    fprintf(stderr, "[nk] I have received an FIR kernel, but the devs are lazy and haven't implemented that yet!\n");
    fprintf(stderr, "[nk] The dummy runtime needs to be provided an implementation in order to execute this task\n\n");
    exit(1);
  } else if (kernel_str == "DASH_SpectralOpening") {
    fprintf(stderr, "[nk] I have received a Spectral Opening kernel, but the devs are lazy and haven't implemented that yet!\n");
    fprintf(stderr, "[nk] The dummy runtime needs to be provided an implementation in order to execute this task\n\n");
    exit(1);
  } else if (kernel_str == "DASH_CIC0") {
    fprintf(stderr, "[nk] I have received a CIC0 kernel, but the devs are lazy and haven't implemented that yet!\n");
    fprintf(stderr, "[nk] The dummy runtime needs to be provided an implementation in order to execute this task\n\n");
    exit(1);
  } else if (kernel_str == "POISON_PILL") {
    printf("[nk] I am inside the runtime's codebase, injecting a poison pill to tell the host thread that I'm done executing\n");
    va_end(args);

    task_node* new_node = (task_node*) calloc(1, sizeof(task_node));
    new_node->name = "POISON PILL";

    task_list.push(new_node);
    printf("[nk] I have pushed the poison pill onto the task list\n");
  } else {
    printf("[nk] Unrecognized kernel specified! (%s)\n", kernel_name);
    va_end(args);
    exit(1);
  }
}

int main(int argc, char** argv) {
  printf("Launching the main function of the mock 'runtime' thread [cedr].\n\n");

  std::string shared_object_name;
  int appInstances = 1;
  int nbCompletedApps = 0;

  if (argc > 2) {
    shared_object_name = std::string(argv[1]);
    appInstances = atoi(argv[2]);
  } else if (argc > 1){
    shared_object_name = std::string(argv[1]);
  } else {
    shared_object_name = "./child.so";
  }

  void *dlhandle = dlopen(shared_object_name.c_str(), RTLD_LAZY);
  void (*lib_main)(void*);
  if (dlhandle == NULL) {
    fprintf(stderr, "Unable to open child shared object: %s (perhaps prepend './'?)\n", shared_object_name.c_str());
    return -1;
  } 
  lib_main = (void(*)(void*))dlsym(dlhandle, "main");

  if (lib_main == NULL) {
    fprintf(stderr, "Unable to get function handle\n");
    return -1;
  } 

  pthread_t worker_thread[appInstances];
  printf("[cedr] Launching %d instances of the received application!\n", appInstances);
  for (int p = 0; p < appInstances; p++)
    pthread_create(&worker_thread[p], nullptr, (void *(*)(void *))lib_main, nullptr);
  void* args[MAX_ARGS];
  task_node* curr_node;
  while (true) {
    if (task_list.try_pop(curr_node)){
      printf("[cedr] I have a task to do!\n");

      if (curr_node->name == "POISON PILL") {
        printf("[cedr] I received a poison pill task! Time to break out of my loop and die...\n");
        nbCompletedApps++;
        free(curr_node);
        if (nbCompletedApps == appInstances)
          break;
        else
          continue;
      }

      printf("[cedr] It is time for me to process a node named %s\n", curr_node->name.c_str());
      
      if (curr_node->args.size() > MAX_ARGS) {
        fprintf(stderr, "[cedr] This node has too many arguments! I can't run it!\n");
        exit(1);
      }
      for (int i = 0; i < curr_node->args.size(); i++) {
        if (i < curr_node->args.size()) {
          args[i] = curr_node->args.at(i);
        } else {
          args[i] = nullptr;
        }
      }

      void* task_run_func = curr_node->run_function;
      void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
      *reinterpret_cast<void **>(&run_func) = task_run_func;  

      printf("[cedr] Calling the implementation of this node\n");
      (run_func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], 
                 args[10], args[11], args[12], args[13], args[14]);
      
      printf("[cedr] Execution is complete. Triggering barrier so that the other thread continues execution\n");
      pthread_barrier_wait(curr_node->completion_barrier);
      
      printf("[cedr] Barrier finished, going to delete this task node now\n");
      free(curr_node); 
    }
  }

  for (int p = 0; p < appInstances; p++)
    pthread_join(worker_thread[p], nullptr);

  printf("[cedr] The worker thread has joined, shutting down...\n");

  dlclose(dlhandle);
  return 0;
}


