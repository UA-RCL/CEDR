#include "runtime.hpp"
#include "scheduler.hpp"
#include <plog/Log.h>
#include <deque>
#include <queue>
#include <mutex>

#include "ipc.h"
#include "dag_parse.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <list>
#include <thread>

#include <random>
#include <sched.h>
#include <pthread.h>
#include <cstdarg>


// Initialize ready and poison queues
std::deque<task_nodes *> ready_queue;   // INFO: Use for storing DASH_API tasks only <- APPs will push DASH_API calls into this queue
std::deque<task_nodes *> poison_queue;  // NOTE: enqueue_task will push poison-pills only into this queue
pthread_mutex_t ready_queue_mutex;
pthread_mutex_t poison_queue_mutex;

std::map <pthread_t, cedr_app *> app_thread_map;
pthread_mutex_t app_thread_map_mutex;

//std::map <std::string, void * [(uint8_t) resource_type::NUM_RESOURCE_TYPES]> dash_functions; // Can be indexed with "dash_fft" or "dash_mmult".
void* dash_functions [api_types::NUM_API_TYPES][precision_types::NUM_PRECISION_TYPES][resource_type::NUM_RESOURCE_TYPES];

extern "C" void enqueue_kernel(const char* kernel_name, const char* precision_name, uint32_t n_vargs, ...) {
  // Check which type of DASH_API call has been made (if-else)
  // For the selected API call, create list of arguments and push in ready_queue
  // If the poison-pill call has been selected, then push it in poison_queue;

  std::string kernel_str(kernel_name);
  std::string precision_str(precision_name);

  // Special case :0
  // Poison pills are special types of nodes that lack corresponding API calls or API-type enums
  if (kernel_str == "POISON_PILL") {
    LOG_INFO << "I am injecting a poison pill to tell the host thread that I'm done executing";

    task_nodes* new_node = (task_nodes*) calloc(1, sizeof(task_nodes));
    new_node->task_name = "POISON PILL";
    new_node->parent_app_pthread = pthread_self();
    pthread_mutex_lock(&app_thread_map_mutex);
    new_node->app_pnt = app_thread_map[pthread_self()];
    pthread_mutex_unlock(&app_thread_map_mutex);

    pthread_mutex_lock(&poison_queue_mutex);
    poison_queue.push_back(new_node);
    pthread_mutex_unlock(&poison_queue_mutex);
    LOG_INFO <<"I have pushed the poison pill onto the task list";

    return;
  }

  // Other special case :0
  // The user requested running a kernel that we don't recognize
  if (api_types_map.find(kernel_str) == api_types_map.end()) {
    LOG_WARNING << "Received a request to enqueue a kernel of type \'" << kernel_str << "\', but that kernel isn't recognized by the runtime!";
    return;
  }

  api_types api = api_types_map.at(kernel_str);
  precision_types precision = precision_types_map.at(precision_str);

  va_list args;
  va_start(args, n_vargs);

  LOG_INFO << "I am enqueueing a new \'" << kernel_str << "\' task";

  task_nodes* new_node = (task_nodes*) calloc(1, sizeof(task_nodes));
  new_node->task_name = kernel_str;
  new_node->task_type = api;
  
  for (int i = 0; i < n_vargs; i++) {
    // The last varg needs to be our completion barrier struct
    if (i == n_vargs - 1) {
      new_node->kernel_barrier = va_arg(args, cedr_barrier_t*);
    } 
    // Otherwise, just push all the void* args into our list
    else {
      new_node->args.push_back(va_arg(args, void*));
    }
  }

  for (int resource = 0; resource < resource_type::NUM_RESOURCE_TYPES; resource++) {
    // If we have an implementation for kernel "kernel_str" on resource "resource", ...
    if (dash_functions[api][precision][resource] != nullptr) {
      // Add it to our array of function pointers
      new_node->run_funcs[resource] = (void*) dash_functions[api][precision][resource];
      new_node->supported_resources[resource] = true;
    }
  }

  IF_LOG(plog::debug) {
    LOG_DEBUG << "Supported resources for task \'" << kernel_str << "\' are";
    for (const auto resourceType : new_node->supported_resources){
      LOG_DEBUG << resource_type_names[(uint8_t) resourceType];
    }
  }

  new_node->parent_app_pthread = pthread_self();
  //TODO: Does app_thread_map require a mutex?
  pthread_mutex_lock(&app_thread_map_mutex);
  new_node->app_pnt = app_thread_map[pthread_self()];
  pthread_mutex_unlock(&app_thread_map_mutex);
  //new_node->task_id = new_node->app_pnt->task_count;  new_node->app_pnt->task_count++;

  // Push this node onto the ready queue
  // Note: this would be a GREAT place for a lock-free multi-producer queue
  // Otherwise, every application trying to push in new work is going to get stuck waiting for some eventual ready queue mutex
  pthread_mutex_lock(&ready_queue_mutex);
  ready_queue.push_back(new_node);
  pthread_mutex_unlock(&ready_queue_mutex);
  LOG_INFO << "I have finished initializing my \'" << kernel_str << "\' node and pushed it onto the task list";
}

void nk_thread_func(void* runfunc) {
  // Cast our voidptr argument to be a function pointer #justCThings
  // int main(int argc, char** argv, char** envp)
  int (*libmain)(int, char**, char**) = (int(*)(int, char**, char**)) runfunc;
  // Call the library's main
  const char* argv = "cedr_app";
  (*libmain)(1, (char**) &argv, nullptr);
  // Once it's complete, enqueue a poison pill to signal the runtime
  enqueue_kernel("POISON_PILL", precision_type_names[0], 0);
}

void parseAPIImplementations(ConfigManager &cedr_config) {
  std::string dash_so_path = cedr_config.getDashBinaryPath();
  LOG_INFO << "Attempting to open dash binary at " << dash_so_path;
  
  void *dash_dlhandle = dlopen(dash_so_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  
  if(!dash_dlhandle) {
    fprintf(stderr, "Failed to open DASH library at: %s. Please ensure the shared object is present or adjust your configuration file.\nFurther details are provided below:\n%s\n", dash_so_path.c_str(), dlerror());
    exit(1);
  }

  LOG_INFO << "Dash binary opened successfully. Scanning for API implementations";
  for (int api = 0; api < api_types::NUM_API_TYPES; api++) {
    for (int precision = 0; precision < precision_types::NUM_PRECISION_TYPES; precision++) {
      for (int resource = 0; resource < resource_type::NUM_RESOURCE_TYPES; resource++) {
        std::string api_name(api_type_names[api]);
        std::string precision_name(precision_type_names[precision]);
        std::string resource_name(resource_type_names[resource]);

        std::string expected_api_function = api_name + "_" + precision_name + "_" + resource_name;
        LOG_DEBUG << "Attempting to find API implementation \'" << expected_api_function << "\'";
        void* symbol_ptr = dlsym(dash_dlhandle, expected_api_function.c_str());
        if (symbol_ptr == NULL) {
          LOG_DEBUG << "Unable to locate API implementation \'" << expected_api_function << "\'";
        } else {
          LOG_INFO << "Located API implementation \'" << expected_api_function << "\'";
        }
        // Regardless of the result, we'll assign it to our array and just live with null checks elsewhere
        dash_functions[api][precision][resource] = symbol_ptr;
      }
    }
  }

}

void launchDaemonRuntime(ConfigManager &cedr_config, pthread_t *resource_handle, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex) {
  LOG_DEBUG << "Beginning execution of Performance mode";

  struct timespec real_current_time {};

  struct timespec sleep_time {};
  sleep_time.tv_sec = 0;
  sleep_time.tv_nsec = 2000;
  std::list<struct struct_logging> stream_timing_log;

  // Schedule timing
  struct timespec schedule_starttime {};
  struct timespec schedule_stoptime {};
  std::list<struct struct_schedlogging> sched_log;
  std::list<struct struct_applogging> app_log;

  struct timespec ipc_time_start {}; 
  struct timespec ipc_time_end{};

  LOG_INFO << "Initializing the shared memory region in the server to communicate with the clients";
  struct process_ipc_struct ipc_cmd {};
  ipc_cmd.message = (ipc_message_struct *) calloc(1, sizeof(ipc_message_struct));
  initialize_sh_mem_server(&ipc_cmd);

  // Scan our provided API binary for implementations that can be used for each of our various APIs
  parseAPIImplementations(cedr_config);

  // Construct a min heap-style priority queue where the application with the
  // lowest arrival time is always on top
  auto app_comparator = [](cedr_app *first, cedr_app *second) { return second->arrival_time < first->arrival_time; };
  std::priority_queue<cedr_app *, std::vector<cedr_app *>, decltype(app_comparator)> unarrived_apps(app_comparator);

  std::list<cedr_app *> arrived_nonstreaming_apps;

  int appNum = 0;

  int completed_task_queue_length = 0;

  uint32_t free_resource_count = cedr_config.getTotalResources();

  pthread_mutex_init(&ready_queue_mutex, NULL);
  pthread_mutex_init(&poison_queue_mutex, NULL);

  std::map<std::string, void *> sharedObjectMap;    // NOTE: May not be needed
  std::map<std::string, cedr_app *> applicationMap;

  // Initialize exponential distribution and random number generator
  std::default_random_engine generator(cedr_config.getRandomSeed());
  std::exponential_distribution<double> distribution(1.0f);

  // Create directory for the daemon process to capture the execution time for the submitted jobs
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
  auto shouldExitEarly = [&cedr_config, &receivedFirstSubDag, &unarrived_apps, &arrived_nonstreaming_apps]() {
    return (cedr_config.getExitWhenIdle() && receivedFirstSubDag && unarrived_apps.empty() && arrived_nonstreaming_apps.empty());
  };

  // Build a CPU set that will allow us to assign all NK threads to run on cores other than where CEDR is
  cpu_set_t non_kernel_cpuset;
  const unsigned int processor_count = std::thread::hardware_concurrency();
  CPU_ZERO(&non_kernel_cpuset);
  for (int cpu_id = 1; cpu_id < processor_count; cpu_id++) {
    CPU_SET(cpu_id, &non_kernel_cpuset);
  }

  // Variable to keep track of killed/resolved applications
  int resolved_app_count = 0; 
  int killed_app_count = 0;
  // Begin execution
  LOG_INFO << "Indefinite run starting";
  while (true) {
    sh_mem_server_process_cmd(&ipc_cmd);
    if (ipc_cmd.cmd_type == SH_OBJ) { // Command for new application received
      // clock_gettime(CLOCK_MONOTONIC_RAW, &ipc_time_start);
      char soPaths[IPC_MAX_APPS][IPC_MAX_PATH_LEN];
      uint64_t num_instances[IPC_MAX_APPS];
      uint64_t periods[IPC_MAX_APPS];
      memcpy(soPaths, ipc_cmd.message->path_to_so, sizeof(soPaths));
      memcpy(num_instances, ipc_cmd.message->num_instances, sizeof(num_instances));
      memcpy(periods, ipc_cmd.message->periodicity, sizeof(periods));

      receivedFirstSubDag = true;

      for (auto appIdx = 0; appIdx < IPC_MAX_APPS; appIdx++) {
        LOG_VERBOSE << "Processing Application slot number " << appIdx << " from the IPC cmd";
        if (strlen(soPaths[appIdx]) == 0) {
          LOG_VERBOSE << "This slot was empty, assuming we're done injecting DAGs";
          break;
        }
        const std::string pathToSO(soPaths[appIdx]);
        const uint64_t num_inst = num_instances[appIdx];
        const uint64_t period = periods[appIdx];
        if (num_inst == 0) {
          LOG_WARNING << "APP " << pathToSO << " was injected, but 0 instances were requested, so I'm skipping it";
          continue;
        }

        std::string appName;
        // If we don't have any '/' in the string, assume the DAG name is the full path name
        if (pathToSO.find('/') == std::string::npos) {
          appName = pathToSO;
        } else {
          appName = pathToSO.substr(pathToSO.find_last_of('/') + 1);
        }
        LOG_INFO << "Received application: " << appName << ". Will attempt to inject " << num_inst << " instances of it with a period of " << period << " microseconds";
        cedr_app *prototypeAppPtr;
        auto appIsPresent = applicationMap.find(appName);
        if (appIsPresent == applicationMap.end()) {
          LOG_DEBUG << appName << " is not cached. Parsing and caching";
          prototypeAppPtr = parse_binary(pathToSO, sharedObjectMap);
          if (prototypeAppPtr != nullptr) {
            applicationMap[appName] = prototypeAppPtr;
          }
          else {
            LOG_ERROR << "Failed to open application shared object due to previous errors! Ignoring application " << appName;
          }
        } else {
          LOG_DEBUG << appName << " was cached, initializing with existing prototype";
          prototypeAppPtr = appIsPresent->second;
        }

        uint64_t new_apps_instantiated = 0;
        cedr_app *new_app;
        uint64_t accum_period = period;
        if (period > 0) {
          distribution = std::exponential_distribution<double>(1.0f / period);
        }
        while (prototypeAppPtr && (new_apps_instantiated < num_inst)) {
          clock_gettime(CLOCK_MONOTONIC_RAW, &real_current_time);
          emul_time = (real_current_time.tv_sec * SEC2NANOSEC + real_current_time.tv_nsec);
          new_app = (cedr_app *)calloc(1, sizeof(cedr_app));
          *new_app = *prototypeAppPtr;
          new_app->app_id = appNum;
          new_app->task_count = 0;
          new_app->is_running = false;
          new_app->completed_task_count = 0;
          new_app->arrival_time = emul_time + accum_period * US2NANOSEC;
          if (cedr_config.getFixedPeriodicInjection()) {
            accum_period += period;
          } else {
            accum_period += distribution(generator);
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
        while (true){
          pthread_mutex_lock(&(resource_mutex[i]));
          if(hardware_thread_handle[i].todo_task_dequeue.empty() && hardware_thread_handle[i].resource_state == 0){
            pthread_mutex_unlock(&(resource_mutex[i]));
            break;
          }
          pthread_mutex_unlock(&(resource_mutex[i]));
        }
        // terminate other pthreads
        pthread_mutex_lock(&(resource_mutex[i]));
        hardware_thread_handle[i].resource_state = 3;
        pthread_mutex_unlock(&(resource_mutex[i]));
      }
        // NEW: Add something that joins any running application 'main' function pthread
        // TODO: Needs revision in the following for loops to ensure validity
        LOG_INFO << "Remaining applications running are " << arrived_nonstreaming_apps.size();
        // NEW: Join all the threads regardless of weather they have completed or not.
        for (auto task : poison_queue){
          pthread_join(task->parent_app_pthread, NULL);

          // App runtime logging
          clock_gettime(CLOCK_MONOTONIC_RAW, &real_current_time);
          struct struct_applogging applog_obj {};
          strncpy(applog_obj.app_name, task->app_pnt->app_name, 50);
          applog_obj.app_id = task->app_pnt->app_id;
          applog_obj.arrival = task->app_pnt->arrival_time;
          applog_obj.start = task->app_pnt->start_time;
          applog_obj.end = (real_current_time.tv_sec * SEC2NANOSEC + real_current_time.tv_nsec);
          applog_obj.app_runtime = applog_obj.end - applog_obj.start;
          applog_obj.app_lifetime = applog_obj.end - applog_obj.arrival;
          app_log.push_back(applog_obj);

          resolved_app_count++;
          arrived_nonstreaming_apps.remove(task->app_pnt);
          LOG_INFO << "Joined pthread of application " << task->app_pnt->app_name;
        }
        // NEW: Kick out forcibly the arrived_nonstreaming_apps threads
        if (!arrived_nonstreaming_apps.size()){
          LOG_INFO << "At the end, remaining unresolved app number is " << arrived_nonstreaming_apps.size();
          for (auto running_app : arrived_nonstreaming_apps){
            pthread_cancel(running_app->app_pthread);
            killed_app_count++;
            pthread_mutex_lock(&app_thread_map_mutex);
            app_thread_map.erase(running_app->app_pthread);
            pthread_mutex_unlock(&app_thread_map_mutex);
            arrived_nonstreaming_apps.remove(running_app);
          }
        }

      ipc_cmd.cmd_type = NOPE;
      LOG_INFO << "Exit command initiated. Terminating runtime while-loop";
      break;
    }

    // Push the heads nodes of the the newly arrived applications on the ready_queue
    clock_gettime(CLOCK_MONOTONIC_RAW, &real_current_time);
    emul_time = (real_current_time.tv_sec * SEC2NANOSEC + real_current_time.tv_nsec);
    while (!unarrived_apps.empty() && (unarrived_apps.top()->arrival_time <= emul_time)) {
      cedr_app *app_inst = unarrived_apps.top();
      LOG_DEBUG << "Application: (" << app_inst->app_name << ", " << app_inst->app_id << ") arrived at time " << emul_time;
      app_inst->arrival_time = emul_time;
      // numParallelApps++;
      arrived_nonstreaming_apps.push_back(app_inst);

      unarrived_apps.pop();
      //LOG_INFO << "Unarrived apps size: " << unarrived_apps.size();
      //LOG_INFO << "Arrived Non-streaming app queue size " << arrived_nonstreaming_apps.size();
    }

    for (auto app : arrived_nonstreaming_apps){
      // TODO: Pass proper data type to pthread_create as function handle
      if (!app->is_running) {
        LOG_INFO << "[DEBUG_SETAFFINITY] Trying to set the first pthread affinity";
        pthread_attr_t non_kernel_attr;
        pthread_attr_init(&non_kernel_attr);
        //int affinity_check = pthread_attr_setaffinity_np(&non_kernel_attr, sizeof(cpu_set_t) * (cedr_config.getResourceArray()[resource_type::cpu] - 1),
        //                                                   &non_kernel_cpuset);
        int affinity_check = pthread_attr_setaffinity_np(&non_kernel_attr, sizeof(cpu_set_t),
                                                           &non_kernel_cpuset);
        //int affinity_check = pthread_attr_setaffinity_np(&non_kernel_attr, non_kernel_cpusize, non_kernel_cpuset);
        if (affinity_check != 0){
          LOG_ERROR << "Failed to set cpu affinity for nk thread. Returned value from pthread_setaffinity_np is " << affinity_check;
          exit(1);
        }
        pthread_attr_setinheritsched(&non_kernel_attr, PTHREAD_EXPLICIT_SCHED);
        int setschedpolicy_check = pthread_attr_setschedpolicy(&non_kernel_attr, SCHED_OTHER);
        if (setschedpolicy_check != 0){
          LOG_ERROR << "Failed to set scheduling policy for application " << app->app_name << " with app_id " << app->app_id;
        }
        sched_param non_kernel_schedparams;
        non_kernel_schedparams.sched_priority = 0;
        pthread_attr_setschedparam(&non_kernel_attr, &non_kernel_schedparams);

        // NOTE: Finally creating thread
        clock_gettime(CLOCK_MONOTONIC_RAW, &real_current_time);
        pthread_create(&(app->app_pthread), &non_kernel_attr, (void *(*)(void *)) nk_thread_func, (void*) app->main_func_handle);
        app->start_time = (real_current_time.tv_sec * SEC2NANOSEC + real_current_time.tv_nsec);
        app->is_running = true;
        LOG_INFO << "Thread for application " << app->app_name << " launched!";

        pthread_mutex_lock(&app_thread_map_mutex);
        app_thread_map[app->app_pthread] = app; // NEW: Store application structs indexed by their main thread
        pthread_mutex_unlock(&app_thread_map_mutex);
      }
    }

    pthread_mutex_lock(&ready_queue_mutex);     // TODO: Optimize so that ready_queue doesn't remain locked for the entire performScheduling routine
      if (!ready_queue.empty()) {
        LOG_INFO << "Scheduling round found " << ready_queue.size() << " tasks in ready task queue!";
        struct struct_schedlogging schedlog_obj {};
        schedlog_obj.ready_queue_size = ready_queue.size();
        clock_gettime(CLOCK_MONOTONIC_RAW, &schedule_starttime);
        performScheduling(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
        clock_gettime(CLOCK_MONOTONIC_RAW, &schedule_stoptime);
        schedlog_obj.start = schedule_starttime;
        schedlog_obj.end = schedule_stoptime;
        schedlog_obj.scheduling_overhead = (schedule_stoptime.tv_sec * SEC2NANOSEC + schedule_stoptime.tv_nsec) - (schedule_starttime.tv_sec * SEC2NANOSEC + schedule_starttime.tv_nsec);
        sched_log.push_back(schedlog_obj);
        LOG_DEBUG << "Ready queue has " << ready_queue.size() << " number of tasks after launching application threads!";
        pthread_mutex_unlock(&ready_queue_mutex);
    } else{
        pthread_mutex_unlock(&ready_queue_mutex);
    }

    // NEW: Unload tasks from completed_task_queue in order to log them in logging struct queue
    for (int i = 0; i < cedr_config.getTotalResources(); i++){
      pthread_mutex_lock(&(resource_mutex[i]));
      completed_task_queue_length = hardware_thread_handle[i].completed_task_dequeue.size() ;
      pthread_mutex_unlock(&(resource_mutex[i])) ;
      if (!completed_task_queue_length){
        // No tasks in completed task deque, move on to next worker thread
        continue;
      }

      LOG_VERBOSE << "Resource " << i << " has completed tasks, popping them off";
      for (int t = 0; t < completed_task_queue_length; t++){
        pthread_mutex_lock(&(resource_mutex[i]));
        task_nodes *task = hardware_thread_handle[i].completed_task_dequeue.front();
        hardware_thread_handle[i].completed_task_dequeue.pop_front();
        pthread_mutex_unlock(&(resource_mutex[i]));
        // NEW
	if(task->app_pnt){
          task->app_pnt->completed_task_count++;
	}

        struct struct_logging log_obj {};
        log_obj.task_id = task->task_id;
        if(task->app_pnt){
	  log_obj.app_id = task->app_pnt->app_id;
          strcpy(log_obj.app_name, task->app_pnt->app_name);
	}
	else{
	  log_obj.app_id = -1;
          strcpy(log_obj.app_name, "GNURadio");
	}
        strcpy(log_obj.task_name, task->task_name.c_str());
        strcpy(log_obj.assign_resource_name, task->assigned_resource_name.c_str());
        log_obj.start = task->start;
        log_obj.end = task->end;
        stream_timing_log.push_back(log_obj);
        free(task);
      }
    }

    // NEW: Remove applications that are completed and have pushed their poison pills
    pthread_mutex_lock(&poison_queue_mutex);
    for (auto poison_task = poison_queue.begin(), poison_task_end = poison_queue.end(); poison_task != poison_task_end;){
      auto poison_task_erase = poison_task; poison_task++;
      const auto erased_app = (*poison_task_erase)->app_pnt;
      if (erased_app->completed_task_count == erased_app->task_count){  // INFO: completed_task_count indicates number of tasks
                                                                        // recorded in the log file. To avoid segfaults, we want to
                                                                        // erase an application once all of its tasks are logged.
        int J = pthread_join((*poison_task_erase)->parent_app_pthread, NULL);

        // App runtime logging
        clock_gettime(CLOCK_MONOTONIC_RAW, &real_current_time);
        struct struct_applogging applog_obj {};
        strncpy(applog_obj.app_name, erased_app->app_name, 50);
        applog_obj.app_id = erased_app->app_id;
        applog_obj.arrival = erased_app->arrival_time;
        applog_obj.start = erased_app->start_time;
        applog_obj.end = (real_current_time.tv_sec * SEC2NANOSEC + real_current_time.tv_nsec);
        applog_obj.app_runtime = applog_obj.end - applog_obj.start;
        applog_obj.app_lifetime = applog_obj.end - applog_obj.arrival;
        app_log.push_back(applog_obj);

        if (J==0) {
          LOG_VERBOSE << "Joined application (non-kernel) thread with id " << erased_app->app_pthread;
          resolved_app_count++;
          LOG_INFO << "Number of resolved applications: " << resolved_app_count << "; Number of killed applications: " << killed_app_count;
        } else {
          LOG_ERROR << "Failed to join thread with id " << erased_app->app_pthread;
          exit(1);  // TODO: Don't exit, may be handle it better
        }
        //const auto erased_app = (*poison_task_erase)->app_pnt;
        arrived_nonstreaming_apps.remove(erased_app);
        // Removing corresponding pthread map of erased_app
        auto map_remove = app_thread_map.find(erased_app->app_pthread);
        app_thread_map.erase(map_remove);
        free(erased_app);   // TODO: Can we get around without freeing the app? Might interfere if a poison-task is processed before a DASH task from completed_task_queue

        free(*poison_task_erase);
        poison_queue.erase(poison_task_erase);
      }
    }
    pthread_mutex_unlock(&poison_queue_mutex);

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
                "app_id: %d, app_name: %s, task_id: %d, task_name: %s, "
                "resource_name: %s, ref_start_time: %lld, ref_stop_time: %lld, "
                "actual_exe_time: %lld\n",
                task.app_id, task.app_name, task.task_id, task.task_name, task.assign_resource_name, s0, e0, e0 - s0);

        it = stream_timing_log.erase(it);
      }
    }
    fclose(trace_fp);
    chmod((log_path + "/timing_trace.log").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }

  
  // Schedule log capturing
  {
    std::list<struct struct_schedlogging>::iterator it;
    it = sched_log.begin();

    FILE *schedtrace_fp;
    schedtrace_fp = fopen((log_path + "/schedule_trace.log").c_str(), "w");
    if (schedtrace_fp == nullptr){
      LOG_ERROR << "Error outputting schedule trace file!";
    }
    else{
      long long total_sched_overhead = 0;
      unsigned int total_ready_tasks = 0;
      while (it != sched_log.end()){
        struct struct_schedlogging schedlog_element = *it;
        long long s1 = schedlog_element.start.tv_sec * SEC2NANOSEC + schedlog_element.start.tv_nsec;
        long long e1 = schedlog_element.end.tv_sec * SEC2NANOSEC + schedlog_element.end.tv_nsec;

        fprintf(schedtrace_fp, "ready_queue_size: %u, ref_start_time: %lld, ref_stop_time: %lld, sheduling_overhead: %lld ns\n",
                schedlog_element.ready_queue_size, s1, e1, e1-s1);
        total_sched_overhead += (e1-s1);
        total_ready_tasks += schedlog_element.ready_queue_size;
        it = sched_log.erase(it);
      }
      fprintf(schedtrace_fp, "total_ready_tasks: %u, total_scheduling_overhead: %lld ns",
              total_ready_tasks, total_sched_overhead);
      fclose(schedtrace_fp);
      chmod((log_path + "/schedule_trace.log").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
  }

  // App Runtime log capturing
  {
    std::list<struct struct_applogging>::iterator it;
    it = app_log.begin();

    FILE *apptrace_fp;
    apptrace_fp = fopen((log_path + "/appruntime_trace.log").c_str(), "w");
    if (apptrace_fp == nullptr){
      LOG_ERROR << "Error outputting schedule trace file!";
    }
    else{
      while (it != app_log.end()){
        struct struct_applogging applog_element = *it;

        fprintf(apptrace_fp, "app_id: %d, app_name: %s, ref_arrival_time: %lld, ref_start_time: %lld, ref_end_time: %lld, app_runtime: %lld, app_lifetime: %lld \n",
                applog_element.app_id, applog_element.app_name, applog_element.arrival, applog_element.start, applog_element.end, applog_element.app_runtime, applog_element.app_lifetime);
        it = app_log.erase(it);
      }
      fclose(apptrace_fp);
      chmod((log_path + "/appruntime_trace.log").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
  }

  free(ipc_cmd.message);
  LOG_INFO << "Run logs are available in : " << log_path;
  LOG_INFO << "The makespan of that log is: " << latestFinish - earliestStart << " ns (" << (latestFinish - earliestStart) / 1000 << " us)";
}
