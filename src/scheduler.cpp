#include "scheduler.hpp"
#include <plog/Log.h>
#include <algorithm>
#include <map>
#include <random>

bool attemptToAssignTaskToPE(task_nodes *task, running_task *thread_handle, pthread_mutex_t *resource_mutex) {
  LOG_VERBOSE << "Attempting to assign task " << std::string(task->task_name) << " to resource "
              << std::string(thread_handle->resource_name);
  pthread_mutex_lock(resource_mutex);
  if (thread_handle->resource_stat == 0) {
    pthread_mutex_unlock(resource_mutex);
    thread_handle->task = task;
    task->running_flag = 1;
    strncpy(task->actual_resource_assign, thread_handle->resource_type, sizeof(task->actual_resource_assign) - 1);
    task->alloc_resource_config_input = thread_handle->resource_config_input;
    strncpy(task->assign_resource_name, thread_handle->resource_name, sizeof(task->assign_resource_name) - 1);

    if (task->run_funcs.find(std::string(thread_handle->resource_type)) == task->run_funcs.end()) {
      LOG_ERROR << "Task " << std::string(task->task_name) << " was assigned to a resource of type "
                << std::string(thread_handle->resource_type)
                << ", but that resource has no function handle entry in this task's function map";
      return false;
    }

    task->actual_run_func = task->run_funcs[std::string(thread_handle->resource_type)];

    //    for (int i = 0; i < task->supported_resource_count; i++) {
    //      if (strcmp(task->supported_resources[i], thread_handle->resource_type) == 0) {
    //        task->actual_run_func = task->run_funcs[i];
    //      }
    //    }
    pthread_mutex_lock(resource_mutex);
    thread_handle->resource_stat = 1;
    pthread_mutex_unlock(resource_mutex);
    LOG_VERBOSE << "Resource was idle, assignment performed";
    return true;
  } else {
    pthread_mutex_unlock(resource_mutex);
    LOG_VERBOSE << "Resource was not idle, assignment not performed";
    return false;
  }
}

int scheduleSimple(std::deque<task_nodes *> &ready_queue, const unsigned int resource_count,
                   const unsigned int free_resource_count, running_task *hardware_thread_handle,
                   pthread_mutex_t *resource_mutex) {

  unsigned int tasks_scheduled = 0;
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    if (free_resource_count == tasks_scheduled) {
      break;
    }
    bool task_allocated = false;
    for (int i = resource_count - 1; i >= 0; i--) {
      for (int j = 0; j < (*itr)->supported_resource_count; j++) {
        if (strcmp((*itr)->supported_resources[j], hardware_thread_handle[i].resource_type) == 0) {
          task_allocated = attemptToAssignTaskToPE((*itr), &hardware_thread_handle[i], &resource_mutex[i]);
          if (task_allocated) {
            tasks_scheduled++;
            break;
          }
        }
      }
      if (task_allocated)
        break;
    }
    if (task_allocated) {
      itr = ready_queue.erase(itr);
    } else {
      ++itr;
    }
  }
  return tasks_scheduled;
}

int scheduleRandom(std::deque<task_nodes *> &ready_queue, const unsigned int resource_count,
                   const unsigned int free_resource_count, running_task *hardware_thread_handle,
                   pthread_mutex_t *resource_mutex) {
  std::shuffle(ready_queue.begin(), ready_queue.end(), std::default_random_engine(RAND_SEED));
  return scheduleSimple(ready_queue, resource_count, free_resource_count, hardware_thread_handle, resource_mutex);
}

int scheduleMET(std::deque<task_nodes *> &ready_queue, const unsigned int resource_count,
                const unsigned int free_resource_count, running_task *hardware_thread_handle,
                pthread_mutex_t *resource_mutex) {
  // TODO: If a task has minimum execution on a resource type that there are no instances of (i.e. FFT on x86), then MET
  // perpetually fails to schedule
  if (free_resource_count == 0) {
    return 0;
  }
  unsigned int tasks_scheduled = 0;

  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    if (free_resource_count == tasks_scheduled) {
      break;
    }
    bool task_allocated = false;

    // Identify the resource type that yields minimum execution time for this
    // task
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
        task_allocated = attemptToAssignTaskToPE((*itr), &hardware_thread_handle[i], &resource_mutex[i]);
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
      if (free_resource_count > 0) {
        LOG_DEBUG << "In MET scheduler, free resource count was above 0, but task " << std::string((*itr)->task_name)
                  << " was not assigned to anything.";
        LOG_DEBUG << "If there are no resources available that match the type that gives it minimum execution time, it "
                     "will never be scheduled!";
      }
    }
  }

  return tasks_scheduled;
}

int scheduleHEFT_RT(std::deque<task_nodes *> &ready_queue, const unsigned int resource_count,
                    const unsigned int free_resource_count, running_task *hardware_thread_handle,
                    pthread_mutex_t *resource_mutex) {

  if (free_resource_count == 0) {
    return 0;
  }
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
    if (free_resource_count == tasks_scheduled) {
      break;
    }
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
        task_allocated = attemptToAssignTaskToPE((*itr), &hardware_thread_handle[i], &resource_mutex[i]);
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

int performScheduling(std::deque<task_nodes *> &ready_queue, unsigned int resource_count,
                      unsigned int free_resource_count, running_task *hardware_thread_handle,
                      pthread_mutex_t *resource_mutex, std::string sched_policy) {
  if (sched_policy == "SIMPLE") {
    return scheduleSimple(ready_queue, resource_count, free_resource_count, hardware_thread_handle, resource_mutex);
  } else if (sched_policy == "RANDOM") {
    return scheduleRandom(ready_queue, resource_count, free_resource_count, hardware_thread_handle, resource_mutex);
  } else if (sched_policy == "MET") {
    return scheduleMET(ready_queue, resource_count, free_resource_count, hardware_thread_handle, resource_mutex);
  } else if (sched_policy == "HEFT_RT") {
    return scheduleHEFT_RT(ready_queue, resource_count, free_resource_count, hardware_thread_handle, resource_mutex);
  } else {
    LOG_FATAL << "Unknown scheduling policy selected! Exiting...";
    exit(1);
  }
}