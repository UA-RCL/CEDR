#pragma once

#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include <plog/Log.h>
#include <nlohmann/json.hpp>

#ifdef ARM
#define CORE_RESOURCE_COUNT 3
#define FFT_RESOURCE_COUNT 1
#else
#define CORE_RESOURCE_COUNT 3
// If we're not building for ARM, there are no FFT accelerators
#define FFT_RESOURCE_COUNT 0
#endif

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
  struct task_nodes_t **succ;
  int succ_count;
  struct task_nodes_t **pred;
  int pred_count;
  char task_name[50];
  int complete_flag;
  int running_flag;

  void *actual_run_func;
  std::map<std::string, void *> run_funcs;
  std::vector<variable *> args;

  struct timespec start, end;
  long long actual_execution_time;
  char actual_resource_assign[25];
  char assign_resource_name[25];

  char supported_resource_count;
  char **supported_resources;
  int alloc_resource_config_input;
  float *estimated_execution;

  int in_ready_queue;

  struct task_nodes_t &operator=(const struct task_nodes_t &other) {
    if (this == &other) {
      return *this;
    }

    task_id = other.task_id;
    app_id = other.app_id;
    // Successor/predecessor arrays handled in dag_app assignment operator
    succ_count = other.succ_count;
    pred_count = other.pred_count;

    strncpy(task_name, other.task_name, 50);
    task_name[49] = '\0';

    complete_flag = other.complete_flag;
    running_flag = other.running_flag;

    run_funcs = std::map<std::string, void *>(other.run_funcs);
    // args handled in dag_app assignment operator

    start = other.start;
    end = other.end;
    actual_execution_time = other.actual_execution_time;
    strncpy(actual_resource_assign, other.actual_resource_assign, 25);
    actual_resource_assign[24] = '\0';

    strncpy(assign_resource_name, other.assign_resource_name, 25);
    assign_resource_name[24] = '\0';

    supported_resource_count = other.supported_resource_count;
    supported_resources = (char **)calloc(supported_resource_count, sizeof(char *));
    for (int i = 0; i < supported_resource_count; i++) {
      char *other_resource = other.supported_resources[i];
      supported_resources[i] = (char *)calloc(strlen(other_resource) + 1, sizeof(char));
      strncpy(supported_resources[i], other_resource, strlen(other_resource));
    }
    alloc_resource_config_input = other.alloc_resource_config_input;
    estimated_execution = (float *)calloc(supported_resource_count, sizeof(float));
    for (int i = 0; i < supported_resource_count; i++) {
      estimated_execution[i] = other.estimated_execution[i];
    }
    return *this;
  }
};
typedef struct task_nodes_t task_nodes;

struct struct_app {
  task_nodes *head_node;
  int app_id;
  int task_count;
  char app_name[50];

  std::map<std::string, variable *> variables;

  unsigned long arrival_time;

  struct struct_app &operator=(const struct struct_app &other) {
    if (this == &other) {
      return *this;
    }
    app_id = other.app_id;
    task_count = other.task_count;
    arrival_time = other.arrival_time;

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
        // The user would have been warned about this during dag parsing, don't spam more warnings here
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
        head_node[i].args.push_back(variables[other.head_node[i].args.at(j)->name]);
      }
    }
    return *this;
  }
};
typedef struct struct_app dag_app;

struct struct_running_task {
  task_nodes *task;
  int resource_stat;
  char resource_name[50];
  char resource_type[50];
  int resource_config_input;
};
typedef struct struct_running_task running_task;

struct struct_pthread_arg {
  running_task *task;
  pthread_mutex_t *thread_lock;
};
typedef struct struct_pthread_arg pthread_arg;
