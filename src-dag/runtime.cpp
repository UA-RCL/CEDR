
#include "runtime.hpp"
#include "scheduler.hpp"
#include <plog/Log.h>
#include <deque>
#include <queue>

#include "ipc.h"
#include "dag_parse.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <list>

#include <random>

bool check_pred_comp(task_nodes *task, stream_mode_t stream_enable) {
  // Non-streaming or single-streaming
  if (stream_enable == stream_mode_t::non_streaming || stream_enable == stream_mode_t::single_buffer) {
    for (int i = 0; i < task->pred_count; i++) {
      if ((task->pred[i]->iter) != (task->iter + 1)) {
        return false;
      }
    }
  }
  // Double-streaming
  else if (stream_enable == stream_mode_t::double_buffer) {
    for (int i = 0; i < task->pred_count; i++) {
      if ((task->pred[i]->iter) == (task->iter)) {
        return false;
      }
    }
  } else if (stream_enable == stream_mode_t::sequential) {
    // for sequential execution, head node of app. has tail node of the app.
    // listed as its predecessor. Head node requires tail node to be at the
    // same iteration before starting new execution. This makes sure a new frame
    // processing by the head node only begins once the previous frame has finished
    // processing by the tail node of app., creating a sequential flow.
    if (task->head_node){
      for (int i = 0; i < task->pred_count; i++){
        if ((task->pred[i]->iter) != (task->iter)){
          return false;
        }
      }
    } else { // Non-head/non-tail nodes follow same logic as single/non-streaming check_pred_comp
      for (int i = 0; i < task->pred_count; i++){
        if ((task->pred[i]->iter) == (task->iter)){
          return false;
        }
      }
    }
  } else {
    LOG_WARNING << "Unknown stream_enable value passed into check_pred_comp: " << stream_enable;
  }
  return true;
}

bool check_succ_comp(task_nodes *task, stream_mode_t stream_enable) {
  // Non-streaming or single-streaming
  if (stream_enable == stream_mode_t::non_streaming || stream_enable == stream_mode_t::single_buffer) {
    for (int i = 0; i < task->succ_count; i++) {
      if (((task->succ[i]->iter) != (task->iter))) {
        return false;
      }
    }
  }
  // Double-streaming
  else if (stream_enable == stream_mode_t::double_buffer) {
    for (int i = 0; i < task->succ_count; i++) {
      if ((long long int)(task->succ[i]->iter) < (long long int)((task->iter) - 1)) {
        return false;
      }
    }
  } else if (stream_enable == stream_mode_t::sequential){
    // for sequential execution, tail node of app. has head node of the app.
    // listed as its successor. Tail node requires head node to have completed one
    // higher iteration than itself to indicate presence of a new frame for the tail node
    // to process. This maintains a sequential flow of execution.
    for (int i = 0; i < task->succ_count; i++) {
      if (task->tail_node) {
        if ((task->succ[i]->iter) != (task->iter + 1)){
          return false;
        }
      } else { // Non-head/non-tail nodes follow same logic as single/non-streaming check_succ_comp
        if ((task->succ[i]->iter) != (task->iter)){
          return false;
        }
      }
    }
  } else {
    LOG_WARNING << "Unknown stream_enable value passed into check_succ_comp: " << stream_enable;
  }
  return true;
}

dag_app *instantiate_app(const std::string &app_name, std::map<std::string, dag_app *> &applicationMap) {
  auto appIsPresent = applicationMap.find(app_name);
  dag_app *result = nullptr;
  if (appIsPresent != applicationMap.end()) {
    LOG_DEBUG << "Instantiating one instance of " << app_name;
    result = (dag_app *)calloc(1, sizeof(dag_app));
    *result = *applicationMap[app_name];
  } else {
    LOG_DEBUG << "Application not found, returning nullptr";
  }
  return result;
}

void launchDaemonRuntime(ConfigManager &cedr_config, pthread_t *resource_handle, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex) {
  LOG_DEBUG << "Beginning execution of Performance mode";

  struct timespec real_current_time {};

  struct timespec sleep_time {};
  sleep_time.tv_sec = 0;
  sleep_time.tv_nsec = 2000;
  std::list<struct struct_logging> stream_timing_log;

  struct timespec ipc_time_start {
  }, ipc_time_end{};

  LOG_INFO << "Initializing the shared memory region in the server to communicate with the clients";
  struct process_ipc_struct ipc_cmd {};
  ipc_cmd.message = (ipc_message_struct *)calloc(1, sizeof(ipc_message_struct));
  initialize_sh_mem_server(&ipc_cmd);

  // Construct a min heap-style priority queue where the application with the
  // lowest arrival time is always on top
  auto app_comparator = [](dag_app *first, dag_app *second) { return second->arrival_time < first->arrival_time; };
  std::priority_queue<dag_app *, std::vector<dag_app *>, decltype(app_comparator)> unarrived_apps(app_comparator);

  std::list<dag_app *> arrived_nonstreaming_apps;
  std::list<dag_app *> arrived_streaming_apps;

  std::list<dag_app *> completed_apps;

  int appNum = 0;

  int completed_task_queue_length;

  uint32_t free_resource_count = cedr_config.getTotalResources();

  // Initialize ready queue
  std::deque<task_nodes *> ready_queue;

  std::map<std::string, void *> sharedObjectMap;
  std::map<std::string, dag_app *> applicationMap;

  // Initialize exponential and random number generator
  std::default_random_engine generator(cedr_config.getRandomSeed());
  std::exponential_distribution<double> distribution(1.0f);

  // Create new directory for the daemon process to capture the execution time for the submitted jobs
  struct stat info {};
  std::string log_path;
  if (stat("./log_dir/", &info) != 0) {
    mkdir("./log_dir/", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    chmod("./log_dir/", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
  {
    int run_id = 0;
    std::string run_folder_name = "./log_dir/experiment";
    while (true) {
      if (stat((run_folder_name + std::to_string(run_id)).c_str(), &info) != 0) {
        mkdir((run_folder_name + std::to_string(run_id)).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        chmod((run_folder_name + std::to_string(run_id)).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        log_path = run_folder_name + std::to_string(run_id);
        break;
      }
      run_id++;
    }
    LOG_INFO << "Run logs will be created in : " << log_path;
  }

  long long emul_time;
  bool receivedFirstSubDag = false;
  // Bind all the parameters to this lambda from their outside references such that, when we call the lambda, we don't
  // need a million arguments
  auto shouldExitEarly = [&cedr_config, &receivedFirstSubDag, &unarrived_apps, &arrived_streaming_apps, &arrived_nonstreaming_apps, &completed_apps]() {
    return (cedr_config.getExitWhenIdle() && receivedFirstSubDag && unarrived_apps.empty() && arrived_streaming_apps.empty() && arrived_nonstreaming_apps.empty() &&
            completed_apps.empty());
  };
  // unsigned long numParallelApps = 0;

  // Begin execution
  LOG_INFO << "Indefinite run starting";
  while (true) {
    sh_mem_server_process_cmd(&ipc_cmd);
    if (ipc_cmd.cmd_type == DAG_OBJ) { // Command for new application received
      // clock_gettime(CLOCK_MONOTONIC_RAW, &ipc_time_start);
      char dagPaths[IPC_MAX_DAGS][IPC_MAX_PATH_LEN];
      uint64_t num_instances[IPC_MAX_DAGS];
      uint64_t periods[IPC_MAX_DAGS];
      memcpy(dagPaths, ipc_cmd.message->path_to_dag, sizeof(dagPaths));
      memcpy(num_instances, ipc_cmd.message->num_instances, sizeof(num_instances));
      memcpy(periods, ipc_cmd.message->periodicity, sizeof(periods));

      receivedFirstSubDag = true;

      for (auto dagIdx = 0; dagIdx < IPC_MAX_DAGS; dagIdx++) {
        LOG_VERBOSE << "Processing DAG slot number " << dagIdx << " from the IPC cmd";
        if (strlen(dagPaths[dagIdx]) == 0) {
          LOG_VERBOSE << "This slot was empty, assuming we're done injecting DAGs";
          break;
        }
        const std::string pathToDag(dagPaths[dagIdx]);
        const uint64_t num_inst = num_instances[dagIdx];
        const uint64_t period = periods[dagIdx];
        if (num_inst == 0) {
          LOG_WARNING << "DAG " << pathToDag << " was injected, but 0 instances were requested, so I'm skipping it";
          continue;
        }

        std::string dagName;
        // If we don't have any '/' in the string, assume the DAG name is the full path name
        if (pathToDag.find('/') == std::string::npos) {
          dagName = pathToDag;
        } else {
          dagName = pathToDag.substr(pathToDag.find_last_of('/') + 1);
        }
        LOG_INFO << "Received application: " << dagName << ". Will attempt to inject " << num_inst << " instances of it with a period of " << period << " microseconds";
        dag_app *prototypeAppPtr;
        auto appIsPresent = applicationMap.find(dagName);
        if (appIsPresent == applicationMap.end()) {
          LOG_DEBUG << dagName << " is not cached. Parsing and caching";
          prototypeAppPtr = parse_dag_and_binary(pathToDag, sharedObjectMap);
          applicationMap[dagName] = prototypeAppPtr;
        } else {
          LOG_DEBUG << dagName << " was cached, initializing with existing prototype";
          prototypeAppPtr = appIsPresent->second;
        }

        uint64_t new_apps_instantiated = 0;
        dag_app *new_app;
        uint64_t accum_period = period;
        if (period > 0) {
          distribution = std::exponential_distribution<double>(1.0f / period);
        }
        while (new_apps_instantiated < num_inst) {
          clock_gettime(CLOCK_MONOTONIC_RAW, &real_current_time);
          emul_time = (real_current_time.tv_sec * SEC2NANOSEC + real_current_time.tv_nsec);
          new_app = (dag_app *)calloc(1, sizeof(dag_app));
          *new_app = *prototypeAppPtr;
          new_app->app_id = appNum;
          new_app->completed_task_count = 0;
          new_app->arrival_time = emul_time + accum_period * US2NANOSEC;
          if (cedr_config.getFixedPeriodicInjection()) {
            accum_period += period;
          } else {
            accum_period += distribution(generator);
          }

          for (int i = 0; i < new_app->task_count; i++) {
            new_app->head_node[i].app_id = appNum;
            new_app->head_node[i].iter = 0;
            new_app->head_node[i].complete_flag = false;
            new_app->head_node[i].in_ready_queue = false;
            new_app->head_node[i].running_flag = false;
          }
          appNum++;

          unarrived_apps.push(new_app);

          new_apps_instantiated++;
        }
      }

      ipc_cmd.cmd_type = NOPE;
      // clock_gettime(CLOCK_MONOTONIC_RAW, &ipc_time_end);
      // LOG_WARNING << "Injecting new applications took " << (ipc_time_end.tv_sec - ipc_time_start.tv_sec) *
      // SEC2NANOSEC + (ipc_time_end.tv_nsec - ipc_time_start.tv_nsec) << " nanoseconds";

    } else if (ipc_cmd.cmd_type == SERVER_EXIT || shouldExitEarly()) {
      LOG_INFO << "Command to terminate daemon process received";
      for (int i = 0; i < cedr_config.getTotalResources(); i++) {
        // terminate other pthreads
        pthread_mutex_lock(&(resource_mutex[i]));
        hardware_thread_handle[i].resource_state = 3;
        pthread_mutex_unlock(&(resource_mutex[i]));
      }
      ipc_cmd.cmd_type = NOPE;
      LOG_INFO << "Exit command initiated. Terminating runtime while-loop";
      break;
    }

    // Check the completion status of the running tasks
    for (int i = 0; i < cedr_config.getTotalResources(); i++) {
      pthread_mutex_lock(&(resource_mutex[i]));
      completed_task_queue_length = hardware_thread_handle[i].completed_task_dequeue.size();
      pthread_mutex_unlock(&(resource_mutex[i]));
      // printf("Completed task queue length for [%d] resource is %d\n", i, completed_task_queue_length);
      // fflush(stdout);

      if (!cedr_config.getEnableQueueing() && completed_task_queue_length > 0) {
        free_resource_count++;
      }
      while (completed_task_queue_length > 0) {
        pthread_mutex_lock(&(resource_mutex[i]));
        task_nodes *task = hardware_thread_handle[i].completed_task_dequeue.front();
        hardware_thread_handle[i].completed_task_dequeue.pop_front();
        pthread_mutex_unlock(&(resource_mutex[i]));
        task->complete_flag = true;
        task->iter = task->iter + 1;
        task->app_pnt->completed_task_count = task->app_pnt->completed_task_count + 1;
        task->in_ready_queue = false;
        task->running_flag = false;
        if (task->succ_count == 0 || task->tail_node) {
          LOG_INFO << "Completed the processing of input frame " << (task->iter - 1) << "\n";
        }
        struct struct_logging log_obj {};
        log_obj.frame_id = task->iter - 1;
        strcpy(log_obj.app_name, task->app_pnt->app_name);
        log_obj.task_id = task->task_id;
        log_obj.app_id = task->app_id;
        strcpy(log_obj.task_name, task->task_name.c_str());
        strcpy(log_obj.assign_resource_name, task->assigned_resource_name.c_str());
        log_obj.start = task->start;
        log_obj.end = task->end;
        log_obj.prev_idle_time = task->prev_idle_time;
        stream_timing_log.push_back(log_obj);

        if (task->app_pnt->stream_enable == stream_mode_t::non_streaming) {
          // task->app_pnt->completed_task_count = task->app_pnt->completed_task_count + 1;
          for (int j = 0; j < task->succ_count; j++) {
            if (check_pred_comp(task->succ[j], stream_mode_t::non_streaming) && (task->succ[j]->in_ready_queue == 0)) {
              task->succ[j]->in_ready_queue = true;
              ready_queue.push_back(task->succ[j]);
            }
          }
        }

        if (task->app_pnt->stream_enable == stream_mode_t::non_streaming) {
          if (task->app_pnt->completed_task_count == task->app_pnt->task_count) {
            completed_apps.push_back(task->app_pnt);
            arrived_nonstreaming_apps.remove(task->app_pnt);
          }
        } else {
          if (task->app_pnt->completed_task_count == task->app_pnt->task_count * task->app_pnt->max_stream_frame_count) {
            completed_apps.push_back(task->app_pnt);
            arrived_streaming_apps.remove(task->app_pnt);
          }
        }
        // numParallelApps--;

        completed_task_queue_length--;
      }
    }

    // Push the heads nodes of the the newly arrived applications on the ready_queue
    clock_gettime(CLOCK_MONOTONIC_RAW, &real_current_time);
    emul_time = (real_current_time.tv_sec * SEC2NANOSEC + real_current_time.tv_nsec);
    while (!unarrived_apps.empty() && (unarrived_apps.top()->arrival_time <= emul_time)) {
      dag_app *app_inst = unarrived_apps.top();
      LOG_DEBUG << "Application: (" << app_inst->app_name << ", " << app_inst->app_id << ") arrived at time " << emul_time;
      // numParallelApps++;
      for (int i = 0; i < app_inst->task_count; i++) {
        task_nodes *task = &(app_inst->head_node[i]);
        const bool predecessors_complete = check_pred_comp(task, app_inst->stream_enable);
        if ((task->complete_flag == 0) && (predecessors_complete) && (task->running_flag == 0)) {
          LOG_DEBUG << "Task " << std::string(task->task_name) << " has dependencies met to be enqueued";
          // ready_queue.push_front(task);
          task->in_ready_queue = true;
          ready_queue.push_back(task);
        }
      }

      if (app_inst->stream_enable > 0) {
        arrived_streaming_apps.push_back(app_inst);
      } else {
        arrived_nonstreaming_apps.push_back(app_inst);
      }
      unarrived_apps.pop();
      LOG_DEBUG << "Unarrived apps size: " << unarrived_apps.size();
      LOG_DEBUG << "Ready queue size: " << ready_queue.size();
    }

    for (auto app : arrived_streaming_apps) {
      for (int j = 0; j < app->task_count; j++) {
        task_nodes *task = &(app->head_node[j]);
        if ((task->in_ready_queue == 0) && (task->iter < app->max_stream_frame_count)) {
          if (check_pred_comp(task, app->stream_enable) && check_succ_comp(task, app->stream_enable)) {
            // LOG_INFO<<"Pushing "<<task->task_name << " iter "<< task->iter<<"\n";
            // fflush(stdout);
            if (task->pred_count == 0 || task->head_node) {
              LOG_INFO << "Starting the processing of input frame " << task->iter << "\n";
            }
            ready_queue.push_back(task);
            task->in_ready_queue = true;
          }
        }
      }
    }
    performScheduling(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);

    std::list<dag_app *>::iterator it;
    it = completed_apps.begin();
    while (it != completed_apps.end()) {
      free(*it);
      it = completed_apps.erase(it);
      // numParallelApps--;
    }

    // pthread_yield();
  }

  for (int i = 0; i < cedr_config.getTotalResources(); i++) {
    pthread_join(resource_handle[i], nullptr);
  }
  LOG_INFO << "Terminated threads";
  sleep_time.tv_sec = 0;
  sleep_time.tv_nsec = 10000000;
  nanosleep(&sleep_time, nullptr);
  long long earliestStart = std::numeric_limits<long long>::max();
  long long latestFinish = 0;
  {
    // Sort the stream timing log based on start time of tasks
    // This resolves edge cases where tasks are pushed into the stream_timing_log in the "wrong" order because of
    // the order in which CEDR checks for completed applications across all PEs
    stream_timing_log.sort([](struct struct_logging first, struct struct_logging second) {
      return (first.start.tv_nsec + first.start.tv_sec * SEC2NANOSEC) < (second.start.tv_nsec + second.start.tv_sec * SEC2NANOSEC);
    });
    std::list<struct struct_logging>::iterator it;
    it = stream_timing_log.begin();
    FILE *trace_fp = fopen((log_path + "/timing_trace.log").c_str(), "w");
    if (trace_fp == nullptr) {
      LOG_ERROR << "Error opening output trace file!";
    } else {
      while (it != stream_timing_log.end()) {

        struct struct_logging task = *it;
        long long s0, e0;
        s0 = (task.start.tv_sec * SEC2NANOSEC + task.start.tv_nsec);
        e0 = (task.end.tv_sec * SEC2NANOSEC + task.end.tv_nsec);

        if (e0 > latestFinish) {
          latestFinish = e0;
        }
        if (s0 < earliestStart) {
          earliestStart = s0;
        }

        fprintf(trace_fp,
                "app_id: %d, frame_id: %d, app_name: %s, task_id: %d, task_name: %s, "
                "resource_name: %s, ref_start_time: %lld, ref_stop_time: %lld, "
                "actual_exe_time: %lld, scheduling_overhead: %llu\n",
                task.app_id, task.frame_id, task.app_name, task.task_id, task.task_name, task.assign_resource_name, s0, e0, e0 - s0, task.prev_idle_time);

        it = stream_timing_log.erase(it);
      }
    }
    fclose(trace_fp);
    chmod((log_path + "/timing_trace.log").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
  free(ipc_cmd.message);
  LOG_INFO << "Run logs are available in : " << log_path;
  LOG_INFO << "The makespan of that log is: " << latestFinish - earliestStart << " ns (" << (latestFinish - earliestStart) / 1000 << " us)";
}
