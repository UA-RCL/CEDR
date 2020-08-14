#include "scheduler.hpp"
#include <algorithm>
#include <map>
#include <random>

std::map<std::string, int> schedule_cache;

void cacheScheduleDecision(task_nodes *task, int resource_id) {
  auto key = std::string(task->app_pnt->app_name) + "+" + std::string(task->task_name);
  LOG_VERBOSE << "Caching scheduling decision of key " << key << " with resource_id " << resource_id;
  schedule_cache[key] = resource_id;
}

bool attemptToAssignTaskToPE(task_nodes *task, running_task *thread_handle, pthread_mutex_t *resource_mutex, int idx,
                             bool cache_schedules) {
  LOG_VERBOSE << "Assigning task " << std::string(task->task_name) << " to resource "
              << std::string(thread_handle->resource_name);
  if (task->run_funcs.find(std::string(thread_handle->resource_type)) == task->run_funcs.end()) {
    LOG_ERROR << "Task " << std::string(task->task_name) << " was assigned to a resource of type "
              << std::string(thread_handle->resource_type)
              << ", but that resource has no function handle entry in this task's function map";
    return false;
  }
  task->running_flag = 1;
  strncpy(task->actual_resource_assign, thread_handle->resource_type,
          sizeof(task->actual_resource_assign) - 1);                        // Types of resources (CPU, FFT)
  task->alloc_resource_config_input = thread_handle->resource_config_input; // ID of resource (FFT0 or FFT1)
  strncpy(task->assign_resource_name, thread_handle->resource_name,
          sizeof(task->assign_resource_name) - 1); // Actual name (FFT0, CPU1)

  task->actual_run_func = task->run_funcs[std::string(thread_handle->resource_type)];

  pthread_mutex_lock(resource_mutex);
  thread_handle->todo_task_dequeue.push_back(task);
  pthread_mutex_unlock(resource_mutex);

  LOG_VERBOSE << "Task pushed to the to-do queue of the requested resource";
  if (cache_schedules) {
    cacheScheduleDecision(task, idx);
  }
  return true;
}

int scheduleCached(std::deque<task_nodes *> &ready_queue, running_task *hardware_thread_handle,
                   pthread_mutex_t *resource_mutex) {
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
      task_allocated = attemptToAssignTaskToPE((*itr), &hardware_thread_handle[val], &resource_mutex[val], val, false);
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

int scheduleSimple(std::deque<task_nodes *> &ready_queue, const unsigned int resource_count,
                   running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex, bool cache_schedules) {

  // Note: static initialization only happens on the first call, so this isn't actually overwritten with each call
  static unsigned int rand_resource = 0;
  unsigned int tasks_scheduled = 0;
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    bool task_allocated = false;
    for (int i = resource_count - 1; i >= 0; i--) {
      rand_resource = ++rand_resource % resource_count;
      for (int j = 0; j < (*itr)->supported_resource_count; j++) {
        if (strcmp((*itr)->supported_resources[j], hardware_thread_handle[rand_resource].resource_type) == 0) {
          task_allocated = attemptToAssignTaskToPE((*itr), &hardware_thread_handle[rand_resource],
                                                   &resource_mutex[rand_resource], rand_resource, cache_schedules);
          if (task_allocated) {
            tasks_scheduled++;
            itr = ready_queue.erase(itr);
            break;
          }
        }
      }
      if (task_allocated || itr == ready_queue.end()) {
        break;
      }
    }
  }
  return tasks_scheduled;
}

int scheduleRandom(std::deque<task_nodes *> &ready_queue, const unsigned int resource_count,
                   running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex, bool cache_schedules) {
  std::shuffle(ready_queue.begin(), ready_queue.end(), std::default_random_engine(RAND_SEED));
  return scheduleSimple(ready_queue, resource_count, hardware_thread_handle, resource_mutex, cache_schedules);
}

int scheduleMET(std::deque<task_nodes *> &ready_queue, const unsigned int resource_count,
                running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex, bool cache_schedules) {
  // TODO: If a task has minimum execution on a resource type that there are no instances of (i.e. FFT on x86), then MET
  // perpetually fails to schedule
  unsigned int tasks_scheduled = 0;

  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    bool task_allocated = false;
    // Identify the resource type that yields minimum execution time for this task
    std::pair<std::string, float> min_resource = {"", std::numeric_limits<float>::infinity()};
    for (int i = 0; i < (*itr)->supported_resource_count; i++) {
      if ((*itr)->estimated_execution[i] < min_resource.second) {
        min_resource = {(*itr)->supported_resources[i], (*itr)->estimated_execution[i]};
      }
    }

    // Now look for an instance of that resource type and assign to that if it's free
    for (int i = 0; i < resource_count; i++) {
      if (strcmp(min_resource.first.c_str(), hardware_thread_handle[i].resource_type) == 0) {
        task_allocated =
            attemptToAssignTaskToPE((*itr), &hardware_thread_handle[i], &resource_mutex[i], i, cache_schedules);
        if (task_allocated) {
          tasks_scheduled++;
          break;
        }
      }
    }
    if (task_allocated) {
      itr = ready_queue.erase(itr);
    } else {
      ++itr;
      LOG_DEBUG << "In MET scheduler, after checking resource type of all the resources, the task "
                << std::string((*itr)->task_name) << " was not assigned to anything.";
      LOG_DEBUG << "If there are no resources available that match the type that gives it minimum execution time, it "
                   "will never be scheduled!";
    }
  }

  return tasks_scheduled;
}

int scheduleHEFT_RT(std::deque<task_nodes *> &ready_queue, const unsigned int resource_count,
                    running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex, bool cache_schedules) {

  unsigned int tasks_scheduled = 0;

  std::map<task_nodes *, float> avg_execution_map;

  //"Sort by Rank-U"
  std::sort(ready_queue.begin(), ready_queue.end(),
            [&avg_execution_map](task_nodes *first, task_nodes *second) mutable {
              float first_avg_execution = 0, second_avg_execution = 0;
              for (int i = 0; i < first->supported_resource_count; i++) {
                first_avg_execution += first->estimated_execution[i];
              }
              first_avg_execution /= first->supported_resource_count;

              for (int i = 0; i < second->supported_resource_count; i++) {
                second_avg_execution += second->estimated_execution[i];
              }
              second_avg_execution /= second->supported_resource_count;

              avg_execution_map[first] = first_avg_execution;
              avg_execution_map[second] = second_avg_execution;
              return first_avg_execution > second_avg_execution;
            });

  // Schedule in that order on the resource that minimizes EFT
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    bool task_allocated = false;

    // TODO: Flesh this out. Right now, it's just choosing based on MET because
    // (i) We have no insights into communication costs between elements and
    // (ii) We have no insights into when a PE is estimated to become available
    std::pair<std::string, float> min_resource = {"", std::numeric_limits<float>::infinity()};
    for (int i = 0; i < (*itr)->supported_resource_count; i++) {
      if ((*itr)->estimated_execution[i] < min_resource.second) {
        min_resource = {(*itr)->supported_resources[i], (*itr)->estimated_execution[i]};
      }
    }

    // Now look for an instance of that resource type and assign to that if it's
    // free
    for (int i = 0; i < resource_count; i++) {
      if (strcmp(min_resource.first.c_str(), hardware_thread_handle[i].resource_type) == 0) {
        task_allocated =
            attemptToAssignTaskToPE((*itr), &hardware_thread_handle[i], &resource_mutex[i], i, cache_schedules);
        if (task_allocated) {
          tasks_scheduled++;
          break;
        }
      }
    }
    if (task_allocated) {
      itr = ready_queue.erase(itr);
    } else {
      ++itr;
    }
  }

  return tasks_scheduled;
}

void performScheduling(std::deque<task_nodes *> &ready_queue, unsigned int resource_count,
                       running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex, std::string sched_policy,
                       bool cache_schedules) {
  int tasks_scheduled = 0;
  size_t original_ready_queue_size = ready_queue.size();
  if (original_ready_queue_size == 0) {
    return;
  }
  LOG_DEBUG << "Ready queue non-empty, performing task scheduling";
  // Begin by scheduling all cached tasks if requested
  if (cache_schedules) {
    tasks_scheduled = scheduleCached(ready_queue, hardware_thread_handle, resource_mutex);
  }
  // Then schedule whatever is left with the user's chosen scheduler
  if (sched_policy == "SIMPLE") {
    tasks_scheduled +=
        scheduleSimple(ready_queue, resource_count, hardware_thread_handle, resource_mutex, cache_schedules);
  } else if (sched_policy == "RANDOM") {
    tasks_scheduled +=
        scheduleRandom(ready_queue, resource_count, hardware_thread_handle, resource_mutex, cache_schedules);
  } else if (sched_policy == "MET") {
    tasks_scheduled +=
        scheduleMET(ready_queue, resource_count, hardware_thread_handle, resource_mutex, cache_schedules);
  } else if (sched_policy == "HEFT_RT") {
    tasks_scheduled +=
        scheduleHEFT_RT(ready_queue, resource_count, hardware_thread_handle, resource_mutex, cache_schedules);
  } else {
    LOG_FATAL << "Unknown scheduling policy selected! Exiting...";
    exit(1);
  }
  if (tasks_scheduled == 0 && original_ready_queue_size > 0) {
    LOG_WARNING << "During scheduling, no tasks were assigned despite the ready queue having "
                << original_ready_queue_size << " tasks";
  } else {
    LOG_DEBUG << "Scheduled " << tasks_scheduled << " tasks";
  }
}
