#include <vector>
#include <string>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>

#include <fftw3.h>

#include "dash.h"

#define MAX_ARGS 15

//------------------------ Start Kernel Implementations -----------------------//

// TODO: we probably want this to be an actual FFT even in this dummy implementation
void not_really_an_fft(double** input, double** output, int *size, bool *forwardTrans) {
  for (int i = 0; i < *size; i++) {
    (*output)[i] = (*input)[i] * 2.0;
  }
  return;
}


void fftw3f_fft_cpu(double **input_array, double **output_array, int *n_elements, bool *forwardTrans){
    printf("[FFTW3] Entered fftw function\n");
    fftwf_complex *in;
    fftwf_complex *out;
    fftwf_plan p;

    in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (*n_elements));
    out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (*n_elements));
    //printf("[FFTW3] Allocated fftwf_complex type input and output!\n");
    if (*forwardTrans)
    {
      p = fftwf_plan_dft_1d(*n_elements, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    }
    else
    {
      p = fftwf_plan_dft_1d(*n_elements, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    //printf("[FFTW3] Allocated fftw plan!\n");

    for(size_t i = 0; i < 2 * (*n_elements); i+=2)
    {
      in[i/2][0] = (*input_array)[i];
      in[i/2][1] = (*input_array)[i+1];
    }
    //printf("[FFTW3] Copied input array into fftwf_complex data type in!\n");
    fftwf_execute(p);
    //printf("[FFTW3] Completed fft execution!\n");
    for(size_t i = 0; i < 2 * (*n_elements); i+=2)
    {
      (*output_array)[i] = out[i/2][0];
      (*output_array)[i+1] = out[i/2][1];
    }
    //printf("[FFTW3] Copied output array into fftwf_complex data type out!\n");

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);
    return;
}


//------------------------- End Kernel Implementations ------------------------//

struct task_node_t {
  std::string name;
  std::vector<void*> args;
  void* run_function;
  pthread_barrier_t* completion_barrier;
};
typedef struct task_node_t task_node;

std::vector<task_node*> task_list;
pthread_mutex_t task_list_mutex;

extern "C" void enqueue_kernel(const char* kernel_name, ...) {
  std::string kernel_str(kernel_name);

  va_list args;
  va_start(args, kernel_name);

  if (kernel_str == "FFT") {
    printf("[nk] I am inside the runtime's codebase, unpacking my args to enqueue a new FFT task\n");

    double** input = va_arg(args, double**);
    double** output = va_arg(args, double**);
    int* size = va_arg(args, int*);
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
    new_node->run_function = (void*) fftw3f_fft_cpu; //(void*) not_really_an_fft;
    new_node->completion_barrier = barrier; 
    
    printf("[nk] I have finished initializing my FFT node, pushing it onto the task list\n");

    // Push this node onto the ready queue
    // Note: this would be a GREAT place for a lock-free multi-producer queue
    // Otherwise, every application trying to push in new work is going to get stuck waiting for some eventual ready queue mutex
    pthread_mutex_lock(&task_list_mutex);
    task_list.push_back(new_node);
    pthread_mutex_unlock(&task_list_mutex);
    printf("[nk] I have pushed a new task onto the work queue, time to go sleep until it gets scheduled and completed\n");
  } else if (kernel_str == "MMULT") {
    // Unpack args and enqueue an mmult task

  } else if (kernel_str == "POISON_PILL") {
    printf("[nk] I am inside the runtime's codebase, injecting a poison pill to tell the host thread that I'm done executing\n");
    va_end(args);

    task_node* new_node = (task_node*) calloc(1, sizeof(task_node));
    new_node->name = "POISON PILL";

    pthread_mutex_lock(&task_list_mutex);
    task_list.push_back(new_node);
    pthread_mutex_unlock(&task_list_mutex);
    printf("[nk] I have pushed the poison pill onto the task list\n");
  } else {
    printf("[nk] Unrecognized kernel specified!\n");
    va_end(args);
  }
}

int main(int argc, char** argv) {
  printf("Launching the main function of the mock 'runtime' thread [cedr].\n\n");
 
  pthread_mutex_init(&task_list_mutex, NULL);
   
  std::string shared_object_name;
  if (argc > 1) {
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

  pthread_t worker_thread;
  pthread_create(&worker_thread, nullptr, (void *(*)(void *))lib_main, nullptr);
  void* args[MAX_ARGS];

  while (true) {
    pthread_mutex_lock(&task_list_mutex);
    if (!task_list.empty()) {
      printf("[cedr] I have a task to do!\n");
      task_node* curr_node = task_list.front();
      task_list.pop_back();
      pthread_mutex_unlock(&task_list_mutex);

      if (curr_node->name == "POISON PILL") {
        printf("[cedr] I received a poison pill task! Time to break out of my loop and die...\n");
        break;
      }

      printf("[cedr] It is time for me to process a node named %s\n", curr_node->name.c_str());
      // Setup the arguments 
      
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

      // Call the kernel's implementation (pretending that it was decided through scheduling of some kind)
      printf("[cedr] Calling the implementation of this node\n");
      (run_func)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], 
                 args[10], args[11], args[12], args[13], args[14]);
      
      printf("[cedr] Execution is complete. Triggering barrier so that the other thread continues execution\n");
      pthread_barrier_wait(curr_node->completion_barrier);
      
      printf("[cedr] Barrier finished, going to delete this task node now\n");
      free(curr_node); 
    } else {
      pthread_mutex_unlock(&task_list_mutex);
    }
  }

  pthread_join(worker_thread, nullptr);

  printf("[cedr] The worker thread has joined, shutting down...\n");

  dlclose(dlhandle);
  return 0;
}


