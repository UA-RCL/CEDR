#pragma once
//#define _GNU_SOURCE

#include <dlfcn.h>
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <plog/Log.h>
#include <nlohmann/json.hpp>

#define SEC2NANOSEC 1000000000
#define USEC2NANOSEC 1000
#define NANOSEC2USEC 1000

// How are arrays like "estimated execution time" laid out
// and what index do you use to get the value corresponding to a given resource type?
enum api_types {DASH_FFT = 0, DASH_GEMM = 1, DASH_FIR = 2, DASH_SpectralOpening = 3, DASH_CIC = 4, DASH_ZIP = 5, DASH_BPSK = 6, DASH_QAM16 = 7, DASH_CONV_2D = 8, DASH_CONV_1D = 9, NUM_API_TYPES = 10};
static const char *api_type_names[] = {"DASH_FFT", "DASH_GEMM", "DASH_FIR", "DASH_SpectralOpening", "DASH_CIC", "DASH_ZIP", "DASH_BPSK", "DASH_QAM16", "DASH_CONV_2D", "DASH_CONV_1D"};
static_assert(sizeof(api_type_names) / sizeof(char *) == api_types::NUM_API_TYPES, "API type enum is missing a string representation or enum is missing a value");

static const std::map<std::string, api_types> api_types_map = {{api_type_names[api_types::DASH_FFT], api_types::DASH_FFT},
                                                               {api_type_names[api_types::DASH_GEMM], api_types::DASH_GEMM},
                                                               {api_type_names[api_types::DASH_FIR], api_types::DASH_FIR},
                                                               {api_type_names[api_types::DASH_SpectralOpening], api_types::DASH_SpectralOpening},
                                                               {api_type_names[api_types::DASH_CIC], api_types::DASH_CIC},
                                                               {api_type_names[api_types::DASH_ZIP], api_types::DASH_ZIP},
                                                               {api_type_names[api_types::DASH_BPSK], api_types::DASH_BPSK},
                                                               {api_type_names[api_types::DASH_QAM16], api_types::DASH_QAM16},
                                                               {api_type_names[api_types::DASH_CONV_2D], api_types::DASH_CONV_2D},
                                                               {api_type_names[api_types::DASH_CONV_1D], api_types::DASH_CONV_1D}};
                                                                                                                            

enum precision_types { prec_flt = 0, prec_int = 1, NUM_PRECISION_TYPES = 2 };
static const char *precision_type_names[] = { "flt", "int" };
static_assert(sizeof(precision_type_names) / sizeof(char *) == precision_types::NUM_PRECISION_TYPES, "Precision type enum is missing a string representation or enum is missing a value");

static const std::map<std::string, precision_types> precision_types_map =
                                                              {{precision_type_names[precision_types::prec_flt], precision_types::prec_flt},
                                                               {precision_type_names[precision_types::prec_int], precision_types::prec_int}};

enum resource_type { cpu = 0, fft = 1, mmult = 2, zip = 3, gpu = 4, NUM_RESOURCE_TYPES = 5 };
// https://stackoverflow.com/a/9150607
static const char *resource_type_names[] = {"cpu", "fft", "gemm", "zip", "gpu"};
static_assert(sizeof(resource_type_names) / sizeof(char *) == resource_type::NUM_RESOURCE_TYPES, "Resource type enum is missing a string representation or enum is missing a value");

static const std::map<std::string, resource_type> resource_type_map = {{resource_type_names[(uint8_t) resource_type::cpu], resource_type::cpu},
                                                                       {resource_type_names[(uint8_t) resource_type::fft], resource_type::fft},
                                                                       {resource_type_names[(uint8_t) resource_type::mmult], resource_type::mmult},
                                                                       {resource_type_names[(uint8_t) resource_type::zip], resource_type::zip},
                                                                       {resource_type_names[(uint8_t) resource_type::gpu], resource_type::gpu}};

struct cedr_barrier {
  pthread_cond_t* cond;
  pthread_mutex_t* mutex;
  uint32_t* completion_ctr;
};
typedef struct cedr_barrier cedr_barrier_t;

struct task_node_t {
  std::string task_name;
  api_types task_type;
  int task_id;
  std::vector<void*> args;
  void *run_funcs[(uint8_t) resource_type::NUM_RESOURCE_TYPES] = {};
  cedr_barrier_t *kernel_barrier;
  struct struct_app *app_pnt;   // points to parent app
  bool supported_resources[(uint8_t) resource_type::NUM_RESOURCE_TYPES] = {};
  struct timespec start, end;
  pthread_t parent_app_pthread;
  void *actual_run_func;
  resource_type assigned_resource_type;
  std::string assigned_resource_name;

  struct task_node_t &operator=(const struct task_node_t &other) {
    if (this == &other) {
      return *this;
    }
    task_name = other.task_name;
    task_type = other.task_type;
    args = other.args;
    for (uint8_t i = 0; i < (uint8_t) resource_type::NUM_RESOURCE_TYPES; i++) {
      run_funcs[i] = other.run_funcs[i];
      supported_resources[i] = other.supported_resources[i];
    }

    kernel_barrier = other.kernel_barrier;
    app_pnt = other.app_pnt;
    parent_app_pthread = other.parent_app_pthread;
    actual_run_func = other.actual_run_func;

    return *this;
  }
};
typedef struct task_node_t task_nodes;

struct struct_app {       // TODO: Change to only consist of shared object stuff
  int app_id;
  int task_count;
  char app_name[50];      // TODO: Convert into std::string
  void *dlhandle;
  void (*main_func_handle)(void *);
  bool is_running;
  pthread_t app_pthread;
  unsigned long arrival_time;
  unsigned long start_time;
  int completed_task_count;

  struct struct_app &operator=(const struct struct_app &other) {
    if (this == &other) {
      return *this;
    }
    dlhandle = other.dlhandle;
    main_func_handle = other.main_func_handle;
    strncpy(app_name, other.app_name, 50);
    app_name[49] = '\0';
    arrival_time = other.arrival_time;
    start_time = other.start_time;

    return *this;
  }
};
typedef struct struct_app cedr_app;

// Forward-declaration of ConfigManager class
class ConfigManager;

struct struct_worker_thread {
  std::deque<task_nodes *> todo_task_dequeue;
  std::deque<task_nodes *> completed_task_dequeue;  // TODO: May get rid of this; keeping for logging purpose
  task_nodes *task;
  int resource_state;

  std::string resource_name;
  // What is the index corresponding to this resource type in task-specific arrays like estimated execution time?
  resource_type thread_resource_type;
  // When multiple of a given resource are available, which instance of that resource are we scheduling to?
  unsigned int resource_cluster_idx;
  // Whenever a task is pushed into the todo_task_dequeue, we add its estimated execution to this
  // Whenever that task is completed, we subtract its estimated execution time
  // (Note: NOT the actual execution time. The goal of the subtraction is simply to undo the prior addition)
  long long todo_dequeue_time; // In nanoseconds
  // Whenever a task is launched, we add the todo_dequeue_time to the current time to update the estimated availability
  // time
  // Threads don't update their own avail time, if you need them to update it themself you might need add mutex locks around current usages
  long long thread_avail_time; // In nanoseconds

  ConfigManager *cedr_config;
};
typedef struct struct_worker_thread worker_thread;

struct struct_pthread_arg {
  worker_thread *thread;
  pthread_mutex_t *thread_lock;
};
typedef struct struct_pthread_arg pthread_arg;

struct struct_logging {
  char app_name[50];
  int app_id;
  int task_id;
  char task_name[50];
  char assign_resource_name[25];
  struct timespec start, end;
};

struct struct_schedlogging {
  unsigned int ready_queue_size;
  struct timespec start, end;
  unsigned long long scheduling_overhead;
};

struct struct_applogging {
  char app_name[50];
  int app_id;
  unsigned long long arrival, start, end;
  unsigned long long app_runtime, app_lifetime;
};
