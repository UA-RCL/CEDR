#include "scheduler.hpp"
#include "IL.hpp"
#include <algorithm>
#include <climits>
#include <map>
#include <random>

std::map<std::string, int> schedule_cache;
NN_DATA_ML_SCHED nn_data;

void cacheScheduleDecision(task_nodes *task, int resource_id) {
  auto key = std::string(task->app_pnt->app_name) + "+" + std::string(task->task_name);
  LOG_VERBOSE << "Caching scheduling decision of key " << key << " with resource_id " << resource_id;
  schedule_cache[key] = resource_id;
}

bool attemptToAssignTaskToPE(ConfigManager &cedr_config, task_nodes *task, worker_thread *thread_handle, 
		pthread_mutex_t *resource_mutex, int idx) {
  LOG_VERBOSE << "Attempting to assigning task " << std::string(task->task_name) << " to resource " << std::string(thread_handle->resource_name);
  if (!task->supported_resources[(uint8_t) thread_handle->thread_resource_type]) {
    LOG_INFO << "Scheduler attempted to assign task " << std::string(task->task_name) << " to a resource of type " << resource_type_names[(uint8_t) thread_handle->thread_resource_type]
              << ", but that resource is not supported by this task";
    //LOG_INFO << "Thread resource type of " << thread_handle->resource_name << " is incompatible with supported resources of " << std::string(task->app_pnt->app_name);
    return false;
  } else {
    if (!cedr_config.getEnableQueueing()) { // Non-queued mode
      // std::cout << "[scheduler.cpp] Entered the Non-queued conditional segment" << std::endl;
      pthread_mutex_lock(resource_mutex);
      const auto resource_state = thread_handle->resource_state;
      const auto todo_queue_size = thread_handle->todo_task_dequeue.size();
      const auto comp_queue_size = thread_handle->completed_task_dequeue.size();
      pthread_mutex_unlock(resource_mutex);
      if ((resource_state != 0) || (todo_queue_size > 0) || (comp_queue_size > 0)) { // Thread not idle or has a job lined up
        LOG_VERBOSE << "Resource state is " << resource_state << ", todo queue size is " << todo_queue_size << ", completed queue size is " << comp_queue_size;
        return false;
      }
    }
  }

  task->assigned_resource_type = thread_handle->thread_resource_type;
  task->assigned_resource_name = thread_handle->resource_name;
  task->actual_run_func = task->run_funcs[(uint8_t) thread_handle->thread_resource_type];

  if (cedr_config.getScheduler() == "EFT" || cedr_config.getScheduler() == "ETF" || cedr_config.getScheduler() == "HEFT_RT") {
    struct timespec curr_timespec {};
    clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
    long long curr_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;
    const long long task_exec_ns = cedr_config.getDashExecTime(task->task_type, thread_handle->thread_resource_type);
    const auto avail_time = thread_handle->thread_avail_time;
    thread_handle->thread_avail_time = (curr_time >= avail_time) ? curr_time + task_exec_ns : avail_time + task_exec_ns;
  }

  // Queuing vs. Non-queuing
  pthread_mutex_lock(resource_mutex);
  thread_handle->todo_task_dequeue.push_back(task);
  pthread_mutex_unlock(resource_mutex);


  LOG_INFO << "Task pushed to the to-do queue of the requested resource";
  if (cedr_config.getCacheSchedules()) {
    cacheScheduleDecision(task, idx);
  }
  return true;
}

int scheduleCached(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex) {
  LOG_VERBOSE << "Schedule tasks based on previously cached schedules";
  unsigned int tasks_scheduled = 0;
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    bool task_allocated = false;
    const auto key = std::string((*itr)->app_pnt->app_name) + "+" + std::string((*itr)->task_name);
    LOG_VERBOSE << "Checking if task with key " << key << " is cached";

    if (schedule_cache.find(key) != schedule_cache.end()) {
      LOG_VERBOSE << "Key found, performing resource assignment";
      const auto val = schedule_cache.at(key);
      // Note: we don't need to cache the schedule again, so we pass in "cache schedule == false" here
      task_allocated = attemptToAssignTaskToPE(cedr_config, (*itr), &hardware_thread_handle[val], &resource_mutex[val], val);
    } else {
      LOG_VERBOSE << "Key not found, task remaining unscheduled";
    }

    if (task_allocated) {
      tasks_scheduled++;
      itr = ready_queue.erase(itr);
    } else {
      ++itr;
    }
  }
  return tasks_scheduled;
}


int scheduleSimple(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                   uint32_t &free_resource_count) {
  // Note: static initialization only happens on the first call, so this isn't actually overwritten with each call
  static unsigned int rand_resource = 0;
  unsigned int tasks_scheduled = 0;
  unsigned int total_resources = cedr_config.getTotalResources();
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    bool task_allocated;
    for (int i = 0; i < total_resources; i++) {
      // Just keep trying to assign this task until one works
      task_allocated = attemptToAssignTaskToPE(cedr_config, (*itr), &hardware_thread_handle[rand_resource], &resource_mutex[rand_resource], rand_resource);
      rand_resource = ++rand_resource % cedr_config.getTotalResources();
      if (task_allocated) {
        tasks_scheduled++;
        itr = ready_queue.erase(itr);
        if (!cedr_config.getEnableQueueing()) {
          free_resource_count--;
        }
        break;
      }
    }
    if (!cedr_config.getEnableQueueing() && free_resource_count == 0) {
      break;
    }
    if (!task_allocated) {
      itr++;
    }
  }
  return tasks_scheduled;
}


int scheduleRandom(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                   uint32_t &free_resource_count) {
  std::shuffle(ready_queue.begin(), ready_queue.end(), std::default_random_engine(cedr_config.getRandomSeed()));
  return scheduleSimple(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
}

int scheduleMET(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                uint32_t &free_resource_count) {
  unsigned int tasks_scheduled = 0;

  // We will store the last resource we scheduled to as an index and then start our "search" for a resource of that type
  // from here i.e.: if the first task picks CPU 0, and the next task wants a resource of type "CPU", it will pick CPU 1
  static unsigned int resource_idx = 0;

  unsigned int total_resources = cedr_config.getTotalResources();
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    bool task_allocated = false;
    // Identify the resource type that yields minimum execution time for this task
    std::pair<resource_type, long long> min_resource = {resource_type::cpu, std::numeric_limits<long long>::max()};

    //for (const auto resourceType : (*itr)->supported_resources) {
    for (int i = total_resources - 1; i >= 0; i--) {
      auto resource_type = hardware_thread_handle[i].thread_resource_type;
      const auto estimated_exec = cedr_config.getDashExecTime((*itr)->task_type, resource_type); // (*itr)->estimated_execution[resourceType];
      auto resourceIsSupported = ((*itr)->supported_resources[(uint8_t) resource_type]);
      if (estimated_exec < min_resource.second && resourceIsSupported) {
        min_resource = {resource_type, estimated_exec};
      }
    }

    // Now look for an instance of that resource type and assign to that if it's free
    for (int i = 0; i < total_resources; i++) {
      LOG_VERBOSE << "On the " << i << "-th iteration of looking for a resource in MET, looking at resource " << resource_idx;
      if (min_resource.first == hardware_thread_handle[resource_idx].thread_resource_type) {
        LOG_VERBOSE << "Resource " << resource_idx << " is of the right type, attempting to schedule to it";
        task_allocated = attemptToAssignTaskToPE(cedr_config, (*itr), &hardware_thread_handle[resource_idx], &resource_mutex[resource_idx], (int)resource_idx);
        if (task_allocated) {
          // Use this static int as our resource counter to encourage "load balancing" among the resources
          resource_idx = ++resource_idx % cedr_config.getTotalResources();
          break;
        }
      }
      // Use this static int as our resource counter to encourage "load balancing" among the resources
      resource_idx = ++resource_idx % cedr_config.getTotalResources();
    }
    if (task_allocated) {
      tasks_scheduled++;
      itr = ready_queue.erase(itr);
      if (!cedr_config.getEnableQueueing()) {
        free_resource_count--;
        if (free_resource_count == 0)
          break;
      }
    } else {
      ++itr;
      LOG_DEBUG << "In MET scheduler, after checking resource type of all the resources, the task " << (*itr)->task_name << " was not assigned to anything.";
      LOG_DEBUG << "If there are no resources available that match the type that gives it minimum execution time, it "
                   "will never be scheduled!";
    }
  }

  return tasks_scheduled;
}
int scheduleHEFT_RT(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                    uint32_t &free_resource_count) {

  unsigned int tasks_scheduled = 0;

  //"Sort by Rank-U"
  std::sort(ready_queue.begin(), ready_queue.end(), [&cedr_config](task_nodes *first, task_nodes *second) {
    unsigned long long first_avg_execution = 0, second_avg_execution = 0;
    size_t first_num_resources_supported = 0, second_num_resources_supported = 0;

    for (size_t i = 0; i < resource_type::NUM_RESOURCE_TYPES; i++) {
      if (first->supported_resources[i]) {
        first_avg_execution += cedr_config.getDashExecTime(first->task_type, (resource_type) i);
        first_num_resources_supported++;
      }
      if (second->supported_resources[i]) {
        second_avg_execution += cedr_config.getDashExecTime(second->task_type, (resource_type) i);
        second_num_resources_supported++;
      }
    }
    first_avg_execution /= first_num_resources_supported;
    second_avg_execution /= second_num_resources_supported;

    LOG_VERBOSE << "First task has average execution " << first_avg_execution << " and second task has average execution " << second_avg_execution;
    return first_avg_execution > second_avg_execution;
  });

  IF_LOG(plog::debug) {
    std::string logstr;
    logstr += "The sorted ready queue is given by\n[";
    for (const auto *task : ready_queue) {
      logstr += task->task_name + " ";
    }
    logstr += "]";
    LOG_DEBUG << logstr;
  }

  // Yes, time advances throughout this process, but calling "now" constant will probably lead to less erratic
  // scheduling behavior
  struct timespec curr_timespec {};
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  long long curr_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;

  // Schedule in that sorted order, each task on its resource that minimizes EFT
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    bool task_allocated = false;

    task_nodes *this_task = (*itr);
    api_types this_api_type = this_task->task_type;
    LOG_DEBUG << "Attempting to schedule task " << this_task->task_name << " based on its earliest finish time";
    // Pair that stores the resource index in the hardware threads array along with the current minimum EFT value
    std::pair<int, long long> selected_resource = {-1, std::numeric_limits<long long>::max()};

    // Iterate over all the resources and pick the one with minimum EFT
    // TODO: Note, this currently assumes communication is 0 because we don't have a model for it in CEDR
    for (unsigned int i = 0; i < cedr_config.getTotalResources(); i++) {
      resource_type this_resource_type = hardware_thread_handle[i].thread_resource_type;
      // If we aren't supported on this resource type, skip it
      if (!this_task->supported_resources[this_resource_type]) {
        LOG_VERBOSE << "In HEFT_RT, task " << this_task->task_name << " does not support resource type " << resource_type_names[this_resource_type];
        continue;
      }

      long long ready_time = hardware_thread_handle[i].thread_avail_time;
      long long start_time = (curr_time >= ready_time) ? curr_time : ready_time;
      // TODO: Note, there's currently no concept of a PE queue having a "gap" for the insertion-based EFT to fill
      // TODO: As such, we currently just skip straight to saying "if we execute on this PE, it's going to be by adding
      // to the end"
      long long eft = start_time + cedr_config.getDashExecTime(this_api_type, this_resource_type);
      LOG_VERBOSE << "For resource " << i << ", the calculated EFT was " << eft;
      if (eft < selected_resource.second) {
        LOG_VERBOSE << "This EFT value was smaller than " << selected_resource.second << ", so it is the new EFT";
        selected_resource = {i, eft};
      } else {
        LOG_VERBOSE << "This EFT value was not smaller than " << selected_resource.second << ", so the EFT is unchanged";
      }
    }

    LOG_DEBUG << "The earliest finish time for task " << this_task->task_name << " was given by (resource_id, eft): (" << selected_resource.first << ", "
              << selected_resource.second << ")";

    // Allocate to the resource with minimum EFT
    const auto idx = selected_resource.first;
    task_allocated = attemptToAssignTaskToPE(cedr_config, this_task, &hardware_thread_handle[idx], &resource_mutex[idx], idx);

    if (task_allocated) {
      tasks_scheduled++;
      itr = ready_queue.erase(itr);
    } else {
      LOG_WARNING << "HEFT_RT failed to allocate a task to a PE despite choosing a PE it was compatible with. Moving on to another task and trying again later.";
      ++itr;
    }
  }

  return tasks_scheduled;
}
/*
int scheduleDNN(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                uint32_t &free_resource_count) {

  int pe_idx;
  unsigned int tasks_scheduled = 0;
  bool task_allocated = false;
  // LOG_DEBUG << "Ready_queue size at the beginning of scheduling is  " << ready_queue.size();
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    // LOG_DEBUG << "[SIMPLESCHEDULE] itr of Ready  " << std::string((*itr)->task_name);
    pe_idx = getPrediction(cedr_config, &nn_data, (*itr), hardware_thread_handle, 0);
    // printf("[DEBUG] PE=%d\n",pe_idx);
    // pe_idx = 0;
    // CHECK FOR ACCEL
    if (pe_idx < cedr_config.getTotalResources()) {
      task_allocated = attemptToAssignTaskToPE(cedr_config, (*itr), &hardware_thread_handle[pe_idx], &resource_mutex[pe_idx], pe_idx);
    } else {
      LOG_WARNING << "DNN selected PE with index " << pe_idx << " but that PE does not exist";
    }

    if (task_allocated) {
      tasks_scheduled++;
      itr = ready_queue.erase(itr);
    } else {
      itr++;
    }
  }

  return tasks_scheduled;
}

int scheduleRT(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
               uint32_t &free_resource_count) {

  int pe_idx;
  unsigned int tasks_scheduled = 0;
  bool task_allocated = false;
  // LOG_DEBUG << "Ready_queue size at the beginning of scheduling is  " << ready_queue.size();
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    // LOG_DEBUG << "[SIMPLESCHEDULE] itr of Ready  " << std::string((*itr)->task_name);
    pe_idx = getPrediction(cedr_config, &nn_data, (*itr), hardware_thread_handle, 1) % cedr_config.getTotalResources();

    if (pe_idx < cedr_config.getTotalResources()) {
      task_allocated = attemptToAssignTaskToPE(cedr_config, (*itr), &hardware_thread_handle[pe_idx], &resource_mutex[pe_idx], pe_idx);
    } else {
      LOG_WARNING << "DNN selected PE with index " << pe_idx << " but that PE does not exist";
    }

    if (task_allocated) {
      tasks_scheduled++;
      itr = ready_queue.erase(itr);
    } else {
      itr++;
    }
  }

  return tasks_scheduled;
}
*/
int scheduleEFT(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                uint32_t &free_resource_count) {

  unsigned int tasks_scheduled = 0;
  int eft_resource = 0;
  unsigned long long earliest_estimated_availtime = 0;
  bool task_allocated;

  struct timespec curr_timespec {};
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  long long curr_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;
  long long avail_time;
  long long task_exec_time;

  unsigned int total_resources = cedr_config.getTotalResources();

  // For loop to iterate over all tasks in Ready queue
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    earliest_estimated_availtime = ULLONG_MAX;
    // For each task, iterate over all PE's to find the earliest finishing one
    for (int i = total_resources - 1; i >= 0; i--) {
      auto resourceType = hardware_thread_handle[i].thread_resource_type;
      avail_time = hardware_thread_handle[i].thread_avail_time;
      task_exec_time = cedr_config.getDashExecTime((*itr)->task_type, resourceType);
      auto finishTime = (curr_time >= avail_time) ? curr_time + task_exec_time : avail_time + task_exec_time;
      auto resourceIsSupported = ((*itr)->supported_resources[(uint8_t) resourceType]);
      if (resourceIsSupported && finishTime < earliest_estimated_availtime) {
        earliest_estimated_availtime = finishTime;
        eft_resource = i;
      }
    }

    // Attempt to assign task on earliest finishing PE
    task_allocated = attemptToAssignTaskToPE(cedr_config, (*itr), &hardware_thread_handle[eft_resource], &resource_mutex[eft_resource], eft_resource);

    // If task allocated successfully
    //  1. Increment the number of scheduled tasks
    //  2. Remove the task from ready_queue
    // Else
    //  1. Go to next task in ready_queue
    if (task_allocated) {
      tasks_scheduled++;
      itr = ready_queue.erase(itr);
      if (!cedr_config.getEnableQueueing()) {
        free_resource_count--;
        if (free_resource_count == 0)
          break;
      }
    } else {
      itr++;
    }
  }
  return tasks_scheduled;
}

int scheduleETF(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                uint32_t &free_resource_count) {

  unsigned int tasks_scheduled = 0;
  int etf_resource = 0;
  unsigned long long earliest_estimated_availtime = 0;
  auto minTask = ready_queue.begin();
  int ready_queue_size = ready_queue.size();
  bool task_allocated;

  struct timespec curr_timespec {};
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  long long curr_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;
  long long avail_time;
  long long task_exec_time;

  unsigned int total_resources = cedr_config.getTotalResources();

  for (int t = 0; t < ready_queue_size; t++) { // Should run a maximum iteration of size of the ready queue
    earliest_estimated_availtime = ULLONG_MAX;
    // For loop for going over task list
    for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
      // Run EFT for each task -----------------------------------------
      for (unsigned int i = 0; i < total_resources; i++) {
        auto resourceType = hardware_thread_handle[i].thread_resource_type;
        avail_time = hardware_thread_handle[i].thread_avail_time;
        task_exec_time = cedr_config.getDashExecTime((*itr)->task_type, resourceType);
        auto finishTime = (curr_time >= avail_time) ? curr_time + task_exec_time : avail_time + task_exec_time;

        auto resourceIsSupported = ((*itr)->supported_resources[(uint8_t) resourceType]);

        if (resourceIsSupported && finishTime < earliest_estimated_availtime) {
          earliest_estimated_availtime = finishTime;
          etf_resource = i;
          minTask = itr;
        }
      }
      // End of EFT ----------------------------------------------------
      // Go to next task
      itr++;
    }

    // assign minTask on desired EFT resource
    task_allocated = attemptToAssignTaskToPE(cedr_config, (*minTask), &hardware_thread_handle[etf_resource], &resource_mutex[etf_resource], etf_resource);

    if (task_allocated) {
      tasks_scheduled++;
      minTask = ready_queue.erase(minTask);
      if (!cedr_config.getEnableQueueing()) {
        free_resource_count--;
        if (free_resource_count == 0)
          break;
      }
    } else {
      LOG_DEBUG << "ETF failed to schedule task, tasks remaining in ready_queue " << ready_queue.size() << " \n ";
    }
  }
  return tasks_scheduled;
}


void performScheduling(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                       uint32_t &free_resource_count) {
  int tasks_scheduled = 0;
  size_t original_ready_queue_size = ready_queue.size();
  const std::string &sched_policy = cedr_config.getScheduler();

  if (original_ready_queue_size == 0) {
    printf("empty queue");
	return;
  }
  LOG_DEBUG << "Ready queue non-empty, performing task scheduling";
  //tasks_scheduled += scheduleSimple(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
  // Begin by scheduling all cached tasks if requested
  /*if (cedr_config.getCacheSchedules()) {
    tasks_scheduled = scheduleCached(cedr_config, ready_queue, hardware_thread_handle, resource_mutex);
  }*/
  // Then schedule whatever is left with the user's chosen scheduler
  if (sched_policy == "SIMPLE") {
    tasks_scheduled += scheduleSimple(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
  } else if (sched_policy == "RANDOM") {
    tasks_scheduled += scheduleRandom(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
  } else if (sched_policy == "MET") {
    tasks_scheduled += scheduleMET(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
  } else if (sched_policy == "HEFT_RT") {
    tasks_scheduled += scheduleHEFT_RT(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
  }/* else if (sched_policy == "DNN") {
    tasks_scheduled += scheduleDNN(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
  } else if (sched_policy == "RT") {
    tasks_scheduled += scheduleRT(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
  }*/ else if (sched_policy == "EFT") {
    tasks_scheduled += scheduleEFT(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
  } else if (sched_policy == "ETF") {
    tasks_scheduled += scheduleETF(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
  } else {
    LOG_FATAL << "Unknown scheduling policy selected! Exiting...";
    exit(1);
  }
  if (tasks_scheduled == 0 && original_ready_queue_size > 0) {
    LOG_WARNING << "During scheduling, no tasks were assigned despite the ready queue having " << original_ready_queue_size << " tasks";
	exit(1);
  } else {
    LOG_DEBUG << "Scheduled " << tasks_scheduled << " tasks. There are now " << free_resource_count << " free resources";
  }
}
