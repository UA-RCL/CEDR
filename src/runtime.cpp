
#include "runtime.hpp"
#include "scheduler.hpp"
#include <plog/Log.h>
#include <deque>
#include <queue>
#include <random>

#ifdef DAEMON
#include "ipc.h"
#include "application.hpp"
#include "dag_parse.hpp"
#include <stdio.h>
#include <sys/stat.h>
#include <list>
#endif

bool check_pred_comp(task_nodes *task) {
  for (int i = 0; i < task->pred_count; i++) {
    if (task->pred[i]->complete_flag == 0) {
      return false;
    }
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

void runValidationMode(const unsigned int resource_count, std::map<std::string, dag_app *> &applicationMap,
                       const std::map<std::string, unsigned int> &config_map, pthread_t *resource_handle,
                       running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex, std::string scheduler,
                       bool cache_schedules) {
  LOG_DEBUG << "Beginning execution of Validation mode";

  struct timespec start1, end1;
  struct timespec real_start_time;

  struct timespec sleep_time;
  sleep_time.tv_sec = 0;
  sleep_time.tv_nsec = 2000;
  long long exec_time = 0;

  dag_app *BM_instances;
  int BM_instances_count = 0;

  int completed_task_queue_length;
  int nb_offloaded_task;

  for (auto &kv : config_map) {
    auto appIsPresent = applicationMap.find(kv.first);
    if (appIsPresent != applicationMap.end()) {
      LOG_INFO << "Will enqueue " << kv.second << " instances of " << kv.first;
      BM_instances_count += kv.second;
    } else {
      LOG_WARNING << "Requested to run " << kv.second << " instances of " << kv.first
                  << ", but that application was not loaded";
    }
  }
  BM_instances = (dag_app *)calloc(BM_instances_count, sizeof(dag_app));

  LOG_INFO << "Performing application initialization";
  int appNum = 0;
  for (auto &kv : config_map) {
    auto appIsPresent = applicationMap.find(kv.first);
    if (appIsPresent == applicationMap.end()) {
      // Application is not found, do not perform initialization for it
      LOG_DEBUG << "Skipping initialization of " << kv.first << " as that application wasn't found";
      continue;
    }
    for (int i = 0; i < kv.second; i++) {
      LOG_DEBUG << "Initializing an instance of " << kv.first;
      BM_instances[appNum] = *applicationMap[kv.first];
      BM_instances[appNum].app_id = appNum;
      BM_instances[appNum].arrival_time = 0;
      for (int j = 0; j < BM_instances[appNum].task_count; j++) {
        BM_instances[appNum].head_node[j].app_id = appNum;
      }
      appNum++;
    }
  }

  int ready_queue_len = 0;

  std::deque<task_nodes *> ready_queue;

  LOG_DEBUG << "Initializing ready queue prior to entering the main runtime loop";
  for (int i = 0; i < BM_instances_count; i++) {
    for (int j = 0; j < BM_instances[i].task_count; j++) {
      task_nodes *tmp = &(BM_instances[i].head_node[j]);
      bool predecessors_complete = check_pred_comp(tmp);
      if ((tmp->complete_flag == 0) && (predecessors_complete) && (tmp->running_flag == 0)) {
        LOG_DEBUG << "Task " << std::string(tmp->task_name) << " has dependencies met to be enqueued";
        ready_queue.push_front(tmp);
        ready_queue_len++;
      }
    }
  }
  LOG_DEBUG << "After initialization, the ready queue contains " << ready_queue.size() << " tasks";
  LOG_DEBUG << "Collecting initial timers and entering main runtime loop";
  clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
  clock_gettime(CLOCK_MONOTONIC_RAW, &real_start_time);
  while (true) {
    for (int i = 0; i < resource_count; i++) {
      pthread_mutex_lock(&(resource_mutex[i]));
      completed_task_queue_length =
          hardware_thread_handle[i].completed_task_dequeue.size(); // read completed_task_dequeue_length
      pthread_mutex_unlock(&(resource_mutex[i]));
      nb_offloaded_task = 0; // number of completed tasks updated from completed_task_dequeue
      // To check if completed tasks are properly being pushed and popped out of the completed task dequeue
      // LOG_DEBUG << "Completed task queue length for resource[" << i << "] is " << completed_task_queue_length;
      while (nb_offloaded_task < completed_task_queue_length) {
        pthread_mutex_lock(&(resource_mutex[i]));
        task_nodes *task = hardware_thread_handle[i].completed_task_dequeue.front();
        hardware_thread_handle[i].completed_task_dequeue.pop_front();
        pthread_mutex_unlock(&(resource_mutex[i]));
        task->complete_flag = 1;

        for (int j = 0; j < task->succ_count; j++) { // pushing the successor of completed task to ready queue
          if (check_pred_comp(task->succ[j])) {
            LOG_VERBOSE << "Enqueueing task " << std::string(task->task_name) << " onto the ready queue";
            ready_queue.push_front(task->succ[j]);
            ready_queue_len++;
          }
        }

        ready_queue_len--; // length reduced because of the completed task
        nb_offloaded_task++;
      }
    }
    // Check the exit condition
    if (ready_queue_len == 0) {
      for (int i = 0; i < resource_count; i++) {
        // terminate other pthreads
        pthread_mutex_lock(&(resource_mutex[i]));
        hardware_thread_handle[i].resource_stat = 3; // 3 - safely terminate the pthread
        pthread_mutex_unlock(&(resource_mutex[i]));
      }
      LOG_INFO << "Exit condition met. Terminating runtime while-loop";
      break;
    }

    performScheduling(ready_queue, resource_count, hardware_thread_handle, resource_mutex, scheduler, cache_schedules);
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
  for (int i = 0; i < resource_count; i++) {
    pthread_join(resource_handle[i], nullptr);
  }
  LOG_INFO << "Terminated threads";
  sleep_time.tv_sec = 0;
  sleep_time.tv_nsec = 10000000;
  nanosleep(&sleep_time, nullptr);

  // Trace for Gantt chart
  FILE *trace_fp = fopen("./trace_time.txt", "w");
  if (trace_fp == nullptr) {
    LOG_ERROR << "Error opening output trace file!";
  } else {
    for (int i = 0; i < BM_instances_count; i++) {
      for (int j = 0; j < BM_instances[i].task_count; j++) {
        long long s0, e0;
        s0 = (((long long)BM_instances[i].head_node[j].start.tv_sec * SEC2NANOSEC +
               (long long)BM_instances[i].head_node[j].start.tv_nsec)) -
             ((long long)start1.tv_sec * SEC2NANOSEC + (long long)start1.tv_nsec);
        e0 = (((long long)BM_instances[i].head_node[j].end.tv_sec * SEC2NANOSEC +
               (long long)BM_instances[i].head_node[j].end.tv_nsec)) -
             ((long long)start1.tv_sec * SEC2NANOSEC + (long long)start1.tv_nsec);
        BM_instances[i].head_node[j].actual_execution_time = e0 - s0;
        fprintf(trace_fp,
                "app_id: %d, app_name: %s, task_id: %d, task_name: %s, "
                "resource_name: %s, ref_start_time: %lld, ref_stop_time: %lld, "
                "actual_exe_time: %lld\n",
                BM_instances[i].app_id, BM_instances[i].app_name, BM_instances[i].head_node[j].task_id,
                BM_instances[i].head_node[j].task_name, BM_instances[i].head_node[j].assign_resource_name, s0, e0,
                BM_instances[i].head_node[j].actual_execution_time);
      }
    }
    fclose(trace_fp);
  }

  trace_fp = fopen("./e2e_exe_time.txt", "w");
  if (trace_fp == nullptr) {
    LOG_ERROR << "Error opening output end-to-end execution time file!";
    exit(1);
  } else {
    for (int i = 0; i < BM_instances_count; i++) {
      long long s0, e0;
      s0 = (((long long)BM_instances[i].head_node[0].start.tv_sec * SEC2NANOSEC +
             (long long)BM_instances[i].head_node[0].start.tv_nsec)) -
           ((long long)start1.tv_sec * SEC2NANOSEC + (long long)start1.tv_nsec);
      e0 = (((long long)BM_instances[i].head_node[BM_instances[i].task_count - 1].end.tv_sec * SEC2NANOSEC +
             (long long)BM_instances[i].head_node[BM_instances[i].task_count - 1].end.tv_nsec)) -
           ((long long)start1.tv_sec * SEC2NANOSEC + (long long)start1.tv_nsec);
      fprintf(trace_fp,
              "app_id: %d, app_name: %s, ref_start_time: %lld, ref_stop_time: "
              "%lld, actual_exe_time: %lld\n",
              BM_instances[i].app_id, BM_instances[i].app_name, s0, e0, (e0 - s0));
    }
    fclose(trace_fp);
  }

  exec_time = (((long long)end1.tv_sec * SEC2NANOSEC + (long long)end1.tv_nsec)) -
              (((long long)start1.tv_sec * SEC2NANOSEC + (long long)start1.tv_nsec));
  LOG_INFO << "Execution time (ns): " << exec_time;
}

void runPerformanceMode(const unsigned int resource_count, std::map<std::string, dag_app *> &applicationMap,
                        const std::map<std::string, std::pair<long long, float>> &config_map,
                        pthread_t *resource_handle, running_task *hardware_thread_handle,
                        pthread_mutex_t *resource_mutex, std::string scheduler, bool cache_schedules) {
  LOG_DEBUG << "Beginning execution of Performance mode";

  struct timespec start1, end1;
  struct timespec real_start_time;
  struct timespec real_current_time;

  int completed_task_queue_length;
  int nb_offloaded_task;
  int done_resource;

  struct timespec sleep_time;
  sleep_time.tv_sec = 0;
  sleep_time.tv_nsec = 2000;
  long long exec_time = 0;

  std::vector<dag_app *> app_list;
  // Construct a min heap-style priority queue where the application with the
  // lowest arrival time is always on top
  auto app_comparator = [](dag_app *first, dag_app *second) { return first->arrival_time < second->arrival_time; };
  std::priority_queue<dag_app *, std::vector<dag_app *>, decltype(app_comparator)> unarrived_apps(app_comparator);

  std::default_random_engine rand_eng(RAND_SEED);
  std::uniform_real_distribution<float> unif(0, 1);

  int appNum = 0;
  // Generate applications
  for (auto &kv : config_map) {
    auto appIsPresent = applicationMap.find(kv.first);
    if (appIsPresent != applicationMap.end()) {
      LOG_INFO << "Will generate instances of " << kv.first << " with period " << kv.second.first << " and probability "
               << kv.second.second;
      const long long period = kv.second.first;
      const float probability = kv.second.second;
      for (long long time = 0; time < MAX_RUNTIME; time += period) {
        if (unif(rand_eng) < probability) {
          LOG_DEBUG << "An instance of " << kv.first << " will arrive at time " << time;
          dag_app *app_inst = instantiate_app(kv.first, applicationMap);

          LOG_INFO << "Performing application initialization";
          app_inst->app_id = appNum;
          app_inst->arrival_time = time * MS2NANOSEC; // Elaborate on this one
          for (int i = 0; i < app_inst->task_count; i++) {
            app_inst->head_node[i].app_id = appNum;
          }
          appNum++;

          LOG_DEBUG << "Placing initialized application into global app list "
                       "and unarrived app queue";
          app_list.push_back(app_inst);
          unarrived_apps.push(app_inst);
        }
      }
    } else {
      LOG_WARNING << "Requested to run instances of " << kv.first << ", but that application was not loaded";
    }
  }
  LOG_DEBUG << "In total, " << app_list.size() << " apps were generated";

  // Initialize ready queue
  std::deque<task_nodes *> ready_queue;
  LOG_DEBUG << "Initializing ready queue prior to entering the main runtime loop";
  while (unarrived_apps.top()->arrival_time == 0) {
    dag_app *app_inst = unarrived_apps.top();
    LOG_INFO << "Application: (" << app_inst->app_name << ", " << app_inst->app_id << ") arrived at time 0";
    for (int i = 0; i < app_inst->task_count; i++) {
      task_nodes *task = &(app_inst->head_node[i]);
      const bool predecessors_complete = check_pred_comp(task);
      if ((task->complete_flag == 0) && (predecessors_complete) && (task->running_flag == 0)) {
        LOG_DEBUG << "Task " << std::string(task->task_name) << " has dependencies met to be enqueued";
        ready_queue.push_front(task);
      }
    }
    unarrived_apps.pop();
  }

  LOG_DEBUG << "After initialization, the ready queue contains " << ready_queue.size() << " tasks";
  LOG_DEBUG << "Collecting initial timers and entering main runtime loop";

  clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
  clock_gettime(CLOCK_MONOTONIC_RAW, &real_start_time);
  const long long time_offset = real_start_time.tv_sec * SEC2NANOSEC + real_start_time.tv_nsec;
  long long emul_time = 0;

  // Begin execution
  while (true) {
    // Check the exit condition
    clock_gettime(CLOCK_MONOTONIC_RAW, &real_current_time);
    emul_time = (real_current_time.tv_sec * SEC2NANOSEC + real_current_time.tv_nsec) - time_offset;

    // Condition for terminating the runtime while loop
    if (emul_time >= MAX_RUNTIME * MS2NANOSEC) {
      done_resource = 0; // For counting number of resources that fulfill exit condition
      for (int i = 0; i < resource_count; i++) {
        pthread_mutex_lock(&(resource_mutex[i]));
        if (hardware_thread_handle[i].resource_stat == 0 &&
            hardware_thread_handle[i].completed_task_dequeue.size() == 0 &&
            hardware_thread_handle[i].todo_task_dequeue.size() == 0)
          done_resource++;
        pthread_mutex_unlock(&(resource_mutex[i]));
      }
      if (done_resource == resource_count) { // all the resources have completed execution
        for (int i = 0; i < resource_count; i++) {
          // terminate other pthreads
          pthread_mutex_lock(&(resource_mutex[i]));
          hardware_thread_handle[i].resource_stat = 3;
          pthread_mutex_unlock(&(resource_mutex[i]));
        }
        LOG_INFO << "Exit condition met. Terminating runtime while-loop";
        // printf("Breaking from while loop\n"); fflush(stdout);
        break;
      }
    }
    LOG_VERBOSE << "Emul time: " << emul_time << ", Target runtime: " << MAX_RUNTIME * MS2NANOSEC << " ("
                << (100.0 * emul_time) / (MAX_RUNTIME * MS2NANOSEC) << "%)";

    // Check for completed tasks, mark them as such, free up their resources,
    // and enqueue their dependencies
    for (int i = 0; i < resource_count; i++) {
      pthread_mutex_lock(&(resource_mutex[i]));
      completed_task_queue_length = hardware_thread_handle[i].completed_task_dequeue.size();
      pthread_mutex_unlock(&(resource_mutex[i]));
      nb_offloaded_task = 0;
      while (nb_offloaded_task < completed_task_queue_length) {
        pthread_mutex_lock(&(resource_mutex[i]));
        task_nodes *task = hardware_thread_handle[i].completed_task_dequeue.front();
        hardware_thread_handle[i].completed_task_dequeue.pop_front();
        pthread_mutex_unlock(&(resource_mutex[i]));
        task->complete_flag = 1;

        for (int j = 0; j < task->succ_count; j++) {
          if (check_pred_comp(task->succ[j])) {
            ready_queue.push_front(task->succ[j]);
          }
        }
        nb_offloaded_task++;
      }
    }

    // Check if any tasks have "arrived"
    clock_gettime(CLOCK_MONOTONIC_RAW, &real_current_time);
    emul_time = (real_current_time.tv_sec * SEC2NANOSEC + real_current_time.tv_nsec) - time_offset;
    while (!unarrived_apps.empty() && unarrived_apps.top()->arrival_time <= emul_time) {
      dag_app *app_inst = unarrived_apps.top();
      LOG_DEBUG << "Application: (" << app_inst->app_name << ", " << app_inst->app_id << ") arrived at time "
                << emul_time;
      for (int i = 0; i < app_inst->task_count; i++) {
        task_nodes *task = &(app_inst->head_node[i]);
        const bool predecessors_complete = check_pred_comp(task);
        if ((task->complete_flag == 0) && (predecessors_complete) && (task->running_flag == 0)) {
          LOG_DEBUG << "Task " << std::string(task->task_name) << " has dependencies met to be enqueued";
          ready_queue.push_front(task);
        }
      }
      unarrived_apps.pop();
      LOG_DEBUG << "Unarrived apps size: " << unarrived_apps.size();
    }

    performScheduling(ready_queue, resource_count, hardware_thread_handle, resource_mutex, scheduler, cache_schedules);
  }
  // printf("Getting Clock time\n"); fflush(stdout);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end1);

  // DEBUG- printing status and dequeue lengths of different resources after termination of while loop
  /*for (int i = 0; i < resource_count; i++){
    pthread_mutex_lock(&(resource_mutex[i]));
    printf("The status of resource [%d] is %d, to_do length %zu, completed length %zu\n", i,
  hardware_thread_handle[i].resource_stat, hardware_thread_handle[i].todo_task_dequeue.size(),
  hardware_thread_handle[i].completed_task_dequeue.size()); fflush(stdout); pthread_mutex_unlock(&(resource_mutex[i]));
  }*/

  for (int i = 0; i < resource_count; i++) {
    // printf("Joining thread number [%d]\n", i);  fflush(stdout); //LOG_INFO << "Joining thread number ["<< i << "]";
    pthread_join(resource_handle[i], nullptr);
  }
  LOG_INFO << "Terminated threads";
  sleep_time.tv_sec = 0;
  sleep_time.tv_nsec = 10000000;
  nanosleep(&sleep_time, nullptr);

  // Trace for Gantt chart
  FILE *trace_fp = fopen("./trace_time.txt", "w");
  if (trace_fp == nullptr) {
    LOG_ERROR << "Error opening output trace file!";
  } else {
    for (auto *app : app_list) {
      for (int i = 0; i < app->task_count; i++) {
        long long s0, e0;
        task_nodes task = app->head_node[i];
        s0 = (task.start.tv_sec * SEC2NANOSEC + task.start.tv_nsec) - (start1.tv_sec * SEC2NANOSEC + start1.tv_nsec);
        e0 = (task.end.tv_sec * SEC2NANOSEC + task.end.tv_nsec) - (start1.tv_sec * SEC2NANOSEC + start1.tv_nsec);
        task.actual_execution_time = e0 - s0;
        fprintf(trace_fp,
                "app_id: %d, app_name: %s, task_id: %d, task_name: %s, "
                "resource_name: %s, ref_start_time: %lld, ref_stop_time: %lld, "
                "actual_exe_time: %lld\n",
                app->app_id, app->app_name, task.task_id, task.task_name, task.assign_resource_name, s0, e0,
                task.actual_execution_time);
      }
    }
    fclose(trace_fp);
  }

  trace_fp = fopen("./e2e_exe_time.txt", "w");
  if (trace_fp == nullptr) {
    LOG_ERROR << "Error opening output end-to-end execution time file!";
    exit(1);
  } else {
    for (auto *app : app_list) {
      long long s0, e0;
      const task_nodes first_task = app->head_node[0];
      const task_nodes last_task = app->head_node[app->task_count - 1];
      s0 = (first_task.start.tv_sec * SEC2NANOSEC + first_task.start.tv_nsec) -
           (start1.tv_sec * SEC2NANOSEC + start1.tv_nsec);
      e0 =
          (last_task.end.tv_sec * SEC2NANOSEC + last_task.end.tv_nsec) - (start1.tv_sec * SEC2NANOSEC + start1.tv_nsec);
      fprintf(trace_fp,
              "app_id: %d, app_name: %s, ref_start_time: %lld, ref_stop_time: "
              "%lld, actual_exe_time: %lld\n",
              app->app_id, app->app_name, s0, e0, (e0 - s0));
    }
    fclose(trace_fp);
  }

  exec_time = (((long long)end1.tv_sec * SEC2NANOSEC + (long long)end1.tv_nsec)) -
              (((long long)start1.tv_sec * SEC2NANOSEC + (long long)start1.tv_nsec));
  LOG_INFO << "Execution time (ns): " << exec_time;
}

#ifdef DAEMON
void runPerformanceMode_in_daemon(const unsigned int resource_count, pthread_t *resource_handle,
                                  running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                                  std::string scheduler, bool cache_schedules) {
  LOG_DEBUG << "Beginning execution of Performance mode";

  struct timespec real_current_time;

  struct timespec sleep_time;
  sleep_time.tv_sec = 0;
  sleep_time.tv_nsec = 2000;
  long long exec_time = 0;

  LOG_INFO << "Initializing the shared memory region in the server to communicate with the clients";
  struct process_ipc_struct ipc_cmd;
  initialize_sh_mem_server(&ipc_cmd);

  std::list<dag_app *> app_list;
  // Construct a min heap-style priority queue where the application with the
  // lowest arrival time is always on top
  auto app_comparator = [](dag_app *first, dag_app *second) { return first->arrival_time < second->arrival_time; };
  std::priority_queue<dag_app *, std::vector<dag_app *>, decltype(app_comparator)> unarrived_apps(app_comparator);

  std::default_random_engine rand_eng(RAND_SEED);
  std::uniform_real_distribution<float> unif(0, 1);

  int appNum = 0;

  int completed_task_queue_length;

  // Initialize ready queue
  std::deque<task_nodes *> ready_queue;

  // Create new directory for the daemon process to capture the execution time for the submitted jobs
  struct stat info;
  std::string log_path;
  if (stat("./log_dir/", &info) != 0) {
    mkdir("./log_dir/", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    chmod("./log_dir/", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
  {
    int run_id = 0;
    std::string run_folder_name = "./log_dir/experiment";
    while (1) {
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

  // LOG_DEBUG << "After initialization, the ready queue contains " << ready_queue.size() << " tasks";
  // LOG_DEBUG << "Collecting initial timers and entering main runtime loop";

  long long emul_time = 0;

  // Begin execution
  LOG_INFO << "Indefinite run starting";
  while (true) {
    // Check the exit condition
    clock_gettime(CLOCK_MONOTONIC_RAW, &real_current_time);
    emul_time = (real_current_time.tv_sec * SEC2NANOSEC + real_current_time.tv_nsec);

    sh_mem_server_process_cmd(&ipc_cmd);
    if (ipc_cmd.cmd_type == DAG_OBJ) { // Command for new aplication received
      // std::cout<<"CMD TYPE "<< ipc_cmd.cmd_type<<"\n";
      LOG_INFO << "New application received";
      LOG_INFO << ipc_cmd.path_to_dag;
      LOG_INFO << ipc_cmd.path_to_shared_obj;
      dag_app *new_app;
      new_app = new dag_app();
      LOG_DEBUG << "Initializing handles to shared objects";
      initializeHandles(new_app->sharedObjectMap, ipc_cmd.path_to_shared_obj);
      // initializeHandles_in_daemon(new_app, ipc_cmd.path_to_shared_obj);
      LOG_DEBUG << "Handles initilization complete";
      // initializeApplications(sharedObjectMap, applicationMap, app_dir);
      // std::cout <<"parsing dag"<<"\n";
      LOG_DEBUG << "Parsing DAG for new aplication";
      parse_dag_file(ipc_cmd.path_to_dag, new_app->sharedObjectMap, false, new_app);
      LOG_DEBUG << "Parsing DAG complete";
      new_app->app_id = appNum;
      new_app->completed_task_count = 0;
      new_app->arrival_time = emul_time;
      for (int i = 0; i < new_app->task_count; i++) {
        new_app->head_node[i].app_id = appNum;
      }
      appNum++;

      app_list.push_back(new_app);
      unarrived_apps.push(new_app);
      free(ipc_cmd.path_to_dag);
      free(ipc_cmd.path_to_shared_obj);
      ipc_cmd.cmd_type = NOPE;

    } else if (ipc_cmd.cmd_type == SERVER_EXIT) {
      LOG_INFO << "Command to terminate daemon process received";
      for (int i = 0; i < resource_count; i++) {
        // terminate other pthreads
        pthread_mutex_lock(&(resource_mutex[i]));
        hardware_thread_handle[i].resource_stat = 3;
        pthread_mutex_unlock(&(resource_mutex[i]));
      }
      ipc_cmd.cmd_type = NOPE;
      LOG_INFO << "Exit command initiated. Terminating runtime while-loop";
      break;
    }

    // Check the completion status of the running tasks
    for (int i = 0; i < resource_count; i++) {
      pthread_mutex_lock(&(resource_mutex[i]));
      completed_task_queue_length = hardware_thread_handle[i].completed_task_dequeue.size();
      pthread_mutex_unlock(&(resource_mutex[i]));
      // printf("Completed task queue length for [%d] resource is %d\n", i, completed_task_queue_length);
      // fflush(stdout);
      while (completed_task_queue_length > 0) {
        pthread_mutex_lock(&(resource_mutex[i]));
        task_nodes *task = hardware_thread_handle[i].completed_task_dequeue.front();
        hardware_thread_handle[i].completed_task_dequeue.pop_front();
        pthread_mutex_unlock(&(resource_mutex[i]));
        task->complete_flag = 1;
        task->app_pnt->completed_task_count = task->app_pnt->completed_task_count + 1;

        // Push the new ready tasks on the read_queue
        for (int j = 0; j < task->succ_count; j++) {
          if (check_pred_comp(task->succ[j])) {
            // ready_queue.push_front(task->succ[j]);
            ready_queue.push_back(task->succ[j]);
          }
        }

        completed_task_queue_length--;
      }

      /*if (hardware_thread_handle[i].resource_stat == 2) {
        pthread_mutex_unlock(&(resource_mutex[i]));
        free_resource_count++;
        task_nodes *task = hardware_thread_handle[i].task;
        task->complete_flag = 1;
        task->app_pnt->completed_task_count = task->app_pnt->completed_task_count + 1;
        //Push the new ready tasks on the read_queue
        for (int j = 0; j < task->succ_count; j++) {
          if (check_pred_comp(task->succ[j])) {
            //ready_queue.push_front(task->succ[j]);
            ready_queue.push_back(task->succ[j]);
          }
        }


        pthread_mutex_lock(&(resource_mutex[i]));
        hardware_thread_handle[i].resource_stat = 0;
        hardware_thread_handle[i].task = nullptr;
        pthread_mutex_unlock(&(resource_mutex[i]));
      }*/

      /*else {
          pthread_mutex_unlock(&(resource_mutex[i]));
        }*/
    }

    // Push the heads nodes of the the newly arrived applications on the ready_queue
    clock_gettime(CLOCK_MONOTONIC_RAW, &real_current_time);
    while (!unarrived_apps.empty()) {
      dag_app *app_inst = unarrived_apps.top();
      LOG_DEBUG << "Application: (" << app_inst->app_name << ", " << app_inst->app_id << ") arrived at time "
                << emul_time;
      for (int i = 0; i < app_inst->task_count; i++) {
        task_nodes *task = &(app_inst->head_node[i]);
        const bool predecessors_complete = check_pred_comp(task);
        if ((task->complete_flag == 0) && (predecessors_complete) && (task->running_flag == 0)) {
          LOG_DEBUG << "Task " << std::string(task->task_name) << " has dependencies met to be enqueued";
          // ready_queue.push_front(task);
          ready_queue.push_back(task);
        }
      }
      unarrived_apps.pop();
      LOG_DEBUG << "Unarrived apps size: " << unarrived_apps.size();
      LOG_DEBUG << "Ready queue size: " << ready_queue.size();
    }

    performScheduling(ready_queue, resource_count, hardware_thread_handle, resource_mutex, scheduler, cache_schedules);

    // for(auto *app:)

    {
      std::list<dag_app *>::iterator it;
      it = app_list.begin();
      while (it != app_list.end()) {
        if ((*it)->completed_task_count == (*it)->task_count) {
          FILE *trace_fp = fopen((log_path + "/appnum" + std::to_string((*it)->app_id) + ".log").c_str(), "w");

          if (trace_fp == nullptr) {
            LOG_ERROR << "Error opening output trace file!";
          } else {
            dag_app *app;
            app = *it;
            for (int i = 0; i < app->task_count; i++) {
              long long s0, e0;
              task_nodes task = app->head_node[i];
              s0 = (task.start.tv_sec * SEC2NANOSEC + task.start.tv_nsec);
              e0 = (task.end.tv_sec * SEC2NANOSEC + task.end.tv_nsec);
              task.actual_execution_time = e0 - s0;
              fprintf(trace_fp,
                      "app_id: %d, app_name: %s, task_id: %d, task_name: %s, "
                      "resource_name: %s, ref_start_time: %lld, ref_stop_time: %lld, "
                      "actual_exe_time: %lld\n",
                      app->app_id, app->app_name, task.task_id, task.task_name, task.assign_resource_name, s0, e0,
                      task.actual_execution_time);
            }

            fclose(trace_fp);
            chmod((log_path + "/appnum" + std::to_string((*it)->app_id) + ".log").c_str(),
                  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
          }
          delete (*it);
          it = app_list.erase(it);
        } else {
          it++;
        }
      }
    }
    // pthread_yield();
  }

  for (int i = 0; i < resource_count; i++) {
    pthread_join(resource_handle[i], nullptr);
  }
  LOG_INFO << "Terminated threads";
  sleep_time.tv_sec = 0;
  sleep_time.tv_nsec = 10000000;
  nanosleep(&sleep_time, nullptr);
  LOG_INFO << "Run logs are available in : " << log_path;
}
#endif
