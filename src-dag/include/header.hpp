#pragma once

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
enum class resource_type : uint8_t { cpu = 0, fft = 1, mmult = 2, gpu = 3, NUM_RESOURCE_TYPES = 4 };
// https://stackoverflow.com/a/9150607
static const char *resource_type_names[] = {"cpu", "fft", "mmult", "gpu"};
static_assert(sizeof(resource_type_names) / sizeof(char *) == (uint8_t) resource_type::NUM_RESOURCE_TYPES, "Resource type enum is missing a string representation");

static const std::map<std::string, resource_type> resource_type_map = {{resource_type_names[(uint8_t)resource_type::cpu], resource_type::cpu},
                                                                       {resource_type_names[(uint8_t)resource_type::fft], resource_type::fft},
                                                                       {resource_type_names[(uint8_t)resource_type::mmult], resource_type::mmult},
                                                                       {resource_type_names[(uint8_t)resource_type::gpu], resource_type::gpu}};

enum stream_mode_t { non_streaming = 0, single_buffer = 1, double_buffer = 2, sequential = 3, NUM_STREAM_MODES = 4 };

// https://stackoverflow.com/a/9150607
static const char *stream_mode_names[] = {"none", "single buffer", "double buffer", "sequential"};
static_assert(sizeof(stream_mode_names) / sizeof(char *) == stream_mode_t::NUM_STREAM_MODES, "Stream mode enum is missing a string representation");

static const std::map<std::string, stream_mode_t> stream_mode_map = {
    {stream_mode_names[stream_mode_t::non_streaming], stream_mode_t::non_streaming},
    {stream_mode_names[stream_mode_t::single_buffer], stream_mode_t::single_buffer},
    {stream_mode_names[stream_mode_t::double_buffer], stream_mode_t::double_buffer},
    {stream_mode_names[stream_mode_t::sequential], stream_mode_t::sequential},
};

struct variable_t {
  std::string name;
  unsigned int num_bytes;
  void *heap_ptr;
  bool is_ptr_var;
  unsigned int ptr_alloc_bytes;
};
typedef struct variable_t variable;

struct task_nodes_t {
  int task_id;
  int app_id;
  struct struct_app *app_pnt;

  struct task_nodes_t **succ;
  int succ_count;
  struct task_nodes_t **pred;
  int pred_count;

  std::string task_name;
  unsigned long long int iter;
  bool complete_flag;
  bool in_ready_queue;
  bool running_flag;

  bool head_node;
  bool tail_node;

  // Note: this used to be a map, but it's been changed to a statically allocated array that's functionally equivalent
  // Given that each resource type is pre-assigned to a given index in this array
  // (i.e. entry 0 is either CPU or nullptr, entry 1 is either FFT or nullptr, ...)
  void *run_funcs[(uint8_t)resource_type::NUM_RESOURCE_TYPES];
  std::vector<variable *> args;
  void *actual_run_func;

  struct timespec start, end;
  long long actual_execution_time;
  unsigned long long prev_idle_time;
  resource_type assigned_resource_type;
  std::string assigned_resource_name;

  std::set<resource_type> supported_resources;
  // ID of resource within its cluster of homogeneous resources (i.e. FFT0 or FFT1)
  unsigned int actual_resource_cluster_idx;
  // Array of estimated execution times (in nanoseconds) for this task across each resource type
  // Currently, it's converted from a floating point microsecond-granularity value during JSON parsing
  long long estimated_execution[(uint8_t)resource_type::NUM_RESOURCE_TYPES];

  ~task_nodes_t() {
    free(pred);
    free(succ);
  }

  struct task_nodes_t &operator=(const struct task_nodes_t &other) {
    if (this == &other) {
      return *this;
    }

    task_id = other.task_id;
    app_id = other.app_id;
    // Successor/predecessor arrays handled in dag_app assignment operator
    succ_count = other.succ_count;
    pred_count = other.pred_count;

    task_name = other.task_name;

    head_node = other.head_node;
    tail_node = other.tail_node;

    complete_flag = other.complete_flag;
    running_flag = other.running_flag;

    for (int i = 0; i < (uint8_t)resource_type::NUM_RESOURCE_TYPES; i++) {
      run_funcs[i] = other.run_funcs[i];
    }
    // args handled in dag_app assignment operator

    start = other.start;
    end = other.end;
    actual_execution_time = other.actual_execution_time;
    prev_idle_time = other.prev_idle_time;
    assigned_resource_type = other.assigned_resource_type;
    assigned_resource_name = other.assigned_resource_name;

    supported_resources = std::set<resource_type>(other.supported_resources);
    actual_resource_cluster_idx = other.actual_resource_cluster_idx;
    for (const auto resourceType : supported_resources) {
      estimated_execution[(uint8_t)resourceType] = other.estimated_execution[(uint8_t)resourceType];
    }
    return *this;
  }
};
typedef struct task_nodes_t task_nodes;

struct struct_app {
  task_nodes *head_node;
  int app_id;
  int task_count;
  int completed_task_count;
  stream_mode_t stream_enable;          // if 1 then application is streaming else not streaming
  int max_stream_frame_count; // maximum number of input frames to be processed in streaming manner
  char app_name[50];
  void *dlhandle;

  std::map<std::string, variable *> variables;

  unsigned long arrival_time;

  struct struct_app &operator=(const struct struct_app &other) {
    if (this == &other) {
      return *this;
    }
    app_id = other.app_id;
    task_count = other.task_count;
    arrival_time = other.arrival_time;
    dlhandle = other.dlhandle;

    stream_enable = other.stream_enable;
    max_stream_frame_count = other.max_stream_frame_count;

    strncpy(app_name, other.app_name, 50);
    app_name[49] = '\0';

    variables = std::map<std::string, variable *>();
    for (std::pair<std::string, variable *> element : other.variables) {
      LOG_VERBOSE << "Allocating a copy of variable " << element.first;
      variable *new_var_ptr = (variable *)calloc(1, sizeof(variable));
      new_var_ptr->name = element.first;
      new_var_ptr->num_bytes = element.second->num_bytes;
      new_var_ptr->is_ptr_var = element.second->is_ptr_var;
      new_var_ptr->heap_ptr = calloc(new_var_ptr->num_bytes, sizeof(uint8_t));
      // If a variable was given an initial value in the application JSON, we need to copy it here
      // Additionally, if it is a _pointer_ variable, then we need to allocate new memory for it
      if (element.second->is_ptr_var) {
        // The user would have been warned if this wasn't the case during dag parsing, don't spam more warnings here
        if (sizeof(uint64_t) == new_var_ptr->num_bytes) {
          LOG_VERBOSE << "Allocating a new heap space for pointer var " << std::string(new_var_ptr->name);
          uint64_t *ptr_loc = (uint64_t *)calloc(new_var_ptr->num_bytes, sizeof(uint8_t));
          memcpy(new_var_ptr->heap_ptr, &ptr_loc, new_var_ptr->num_bytes);
        }
      } else {
        memcpy(new_var_ptr->heap_ptr, element.second->heap_ptr, new_var_ptr->num_bytes);
      }
      variables[element.first] = new_var_ptr;
    }

    head_node = (task_nodes *)calloc(task_count, sizeof(task_nodes));
    for (int i = 0; i < task_count; i++) {
      head_node[i] = other.head_node[i];

      head_node[i].pred = (task_nodes **)calloc(other.head_node[i].pred_count, sizeof(task_nodes *));
      head_node[i].succ = (task_nodes **)calloc(other.head_node[i].succ_count, sizeof(task_nodes *));
      for (int j = 0; j < other.head_node[i].pred_count; j++) {
        head_node[i].pred[j] = &head_node[other.head_node[i].pred[j]->task_id];
      }
      for (int j = 0; j < other.head_node[i].succ_count; j++) {
        head_node[i].succ[j] = &head_node[other.head_node[i].succ[j]->task_id];
      }

      head_node[i].args = std::vector<variable *>();
      for (int j = 0; j < other.head_node[i].args.size(); j++) {
        auto varName = other.head_node[i].args.at(j)->name;
        head_node[i].args.push_back(variables[varName]);
      }

      head_node[i].app_pnt = this;
    }
    return *this;
  }
};
typedef struct struct_app dag_app;

// Forward-declaration of ConfigManager class
class ConfigManager;

struct struct_worker_thread {
  std::deque<task_nodes *> todo_task_dequeue;
  std::deque<task_nodes *> completed_task_dequeue;
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
  int frame_id;
  char app_name[50];
  int app_id;
  int task_id;
  char task_name[50];
  char assign_resource_name[25];
  struct timespec start, end;
  unsigned long long prev_idle_time;
};
