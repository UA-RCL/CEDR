#include "threads.hpp"
#include <dlfcn.h>
#include <sched.h>
#include <stdlib.h>
#include <linux/sched.h>
#include <sys/syscall.h>
#include <stdint.h>
#include <thread>
#include <unistd.h>

#define gettid() syscall(__NR_gettid)

#if defined(USEPAPI)
#include "performance_monitor.hpp"
#include <papi/papi.h>
#endif

#include <plog/Log.h>
#include <string>

#define ON_CPU 0
#define ON_ACC 1

struct sched_attr{
  uint32_t size;              /* Size of this structure */
  uint32_t sched_policy;      /* Policy (SCHED_*) */
  uint64_t sched_flags;       /* Flags */
  int32_t sched_nice;         /* Nice value (SCHED_OTHER, SCHED_BATCH) */
  uint32_t sched_priority;    /* Static priority (SCHED_FIFO, SCHED_RR) */
  /* Remaining fields are for SCHED_DEADLINE */
  uint64_t sched_runtime;
  uint64_t sched_deadline;
  uint64_t sched_period;
};

int sched_setattr(pid_t pid, const struct sched_attr *attr, unsigned int flags){
  return syscall(__NR_sched_setattr, pid, attr, flags);
}

void *hardware_thread(void *ptr) {
  pthread_t self = pthread_self();
  pid_t self_tid = gettid();
  clockid_t clock_id;
  if (pthread_getcpuclockid(self, &clock_id) != 0) {
    LOG_FATAL << "Not able to get CLOCK ID";
    exit(1);
  }

  // NEW: Sched_nice approach
  /*LOG_INFO << "Setting the SCHED_NICE values for the spawned hardware thread";
  auto *resource_attr = (sched_attr *)malloc(sizeof(sched_attr));
  {
    (*resource_attr).size = sizeof(sched_attr);
    (*resource_attr).sched_policy = SCHED_OTHER;
    (*resource_attr).sched_flags = 0;
    (*resource_attr).sched_nice = -10;
  }
  pid_t self_pid = gettid();
  LOG_INFO << "Obtained process id of pthread " << self << " as " << self_pid;
  int ret = sched_setattr(self_pid, resource_attr, 0);
  if (ret != 0){
    LOG_ERROR << "Failed to set sched_nice attributes to the hardware thread";
    exit(1);
  }*/
  // NEW: End of Sched_nice approach

  auto *thread_arg = (pthread_arg *)ptr;
  // Since we spend a lot of time dereferencing these pointers from thread_arg, might as well do it once
  auto *worker_thread = thread_arg->thread;
  auto *thread_lock = thread_arg->thread_lock;
  auto *cedr_config = worker_thread->cedr_config;

  void *run_args[MAX_ARGS];
  // long long expected_finish_time;

  // PE idle time variables - [PE level QUEUE stats]
  unsigned long long last_busy = 0, last_avail = 0, total_idle_time = 0;

  int cpu_name = sched_getcpu();
  LOG_DEBUG << "Starting thread " << self << " as resource name " << worker_thread->resource_name << " and type " << resource_type_names[(uint8_t) worker_thread->thread_resource_type]
            << " on cpu id " << cpu_name;

#if defined(USEPAPI)
  int papiRet, papiEventSet = PAPI_NULL;
  long long papiValues[cedr_config->getPAPICounters().size()];

  if (cedr_config->getUsePAPI() && PAPI_is_initialized()) {
    PAPI_create_eventset(&papiEventSet);

    for (const auto &papiEvent : cedr_config->getPAPICounters()) {
      if ((papiRet = PAPI_add_named_event(papiEventSet, papiEvent.c_str())) != PAPI_OK) {
        LOG_ERROR << "PAPI error " << papiRet << ": " << std::string(PAPI_strerror(papiRet));
      }
    }
  }
#endif

  LOG_VERBOSE << "Entering loop for thread " << self;
  while (true) {
    pthread_mutex_lock(thread_lock);
    if ((!worker_thread->todo_task_dequeue.empty())) {
      auto *task = worker_thread->todo_task_dequeue.front();
      worker_thread->todo_task_dequeue.pop_front();
      worker_thread->task = task;
      worker_thread->resource_state = 1;
      const std::vector<void *> &args = task->args;
      void *task_run_func = task->actual_run_func;
      pthread_mutex_unlock(thread_lock);

      void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, 
                       void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
      *reinterpret_cast<void **>(&run_func) = task_run_func;

      // >= here rather than just > because for accelerator-type threads, they need to be able to pass in one extra argument
      // namely, their respective cluster index so that we can leverage multiple FFTs/multiple GEMMs/etc
      if (args.size() >= MAX_ARGS) {
        LOG_ERROR << "Task " << worker_thread->task->task_name << " has too many arguments (" << args.size() << ") in its dispatch call. "
                  << "Please increase the maximum number of arguments supported by the framework and try running again.";
      } else {
        for (unsigned int idx = 0; idx < MAX_ARGS; idx++) {
          if (idx < args.size()) {
            run_args[idx] = task->args.at(idx);
          } else if (idx == args.size() && worker_thread->thread_resource_type != resource_type::cpu) {
            LOG_VERBOSE << "I'm not a CPU-type thread, I'm going to add my cluster index as the last argument";
            run_args[idx] = (void *) worker_thread->resource_cluster_idx;
          }
        }

        LOG_VERBOSE << "About to dispatch " << worker_thread->task->task_name;
#if defined(USEPAPI)
        if (cedr_config->getUsePAPI() && PAPI_is_initialized()) {
          if ((papiRet = PAPI_start(papiEventSet)) != PAPI_OK) {
            LOG_ERROR << "PAPI error " << papiRet << ": " << std::string(PAPI_strerror(papiRet));
          } else {
            LOG_DEBUG << "Started collection of PAPI events";
          }
        }
#endif
        clock_gettime(CLOCK_MONOTONIC_RAW, &(worker_thread->task->start));
        last_busy = (worker_thread->task->start.tv_sec * SEC2NANOSEC) + (worker_thread->task->start.tv_nsec);
        if (last_avail != 0) {
          //worker_thread->task->prev_idle_time = (last_busy - last_avail);
        } else {
          //worker_thread->task->prev_idle_time = 0;
        }

        // TODO: This is currently basically being handled in attemptToAssignTaskToPE. Is that better? worse? idk.
        //  The benefit of having it there is that instantaneous updates of avail time give schedulers like HEFT_RT
        //  fresh info after every scheduling decision But the downside is that the avail time probably skews a bit from
        //  what the actual avail time is
        /*
        expected_finish_time =
            (long long)worker_thread->todo_dequeue_time +
            (long long)worker_thread->task->start.tv_sec*SEC2NANOSEC +
            (long long)worker_thread->task->start.tv_nsec;
        worker_thread->todo_dequeue_time -= task->estimated_execution[worker_thread->resource_type];
        worker_thread->thread_avail_time = expected_finish_time;
        */

        (run_func)(run_args[0], run_args[1], run_args[2], run_args[3], run_args[4], run_args[5], run_args[6],
                   run_args[7], run_args[8], run_args[9], run_args[10], run_args[11], run_args[12], run_args[13], 
                   run_args[14], run_args[15], run_args[16], run_args[17], run_args[18], run_args[19]);
        cedr_barrier_t *barrier = task->kernel_barrier;
        pthread_mutex_lock(barrier->mutex);
        (*(barrier->completion_ctr))++;
        pthread_cond_signal(barrier->cond);
        pthread_mutex_unlock(barrier->mutex);
        clock_gettime(CLOCK_MONOTONIC_RAW, &(worker_thread->task->end));
        last_avail = (worker_thread->task->end.tv_sec * SEC2NANOSEC) + (worker_thread->task->end.tv_nsec);
      }
#if defined(USEPAPI)
      if (cedr_config->getUsePAPI() && PAPI_is_initialized()) {
        if ((papiRet = PAPI_stop(papiEventSet, papiValues)) != PAPI_OK) {
          LOG_ERROR << "PAPI error " << papiRet << ": " << std::string(PAPI_strerror(papiRet));
        } else {
          std::stringstream sstream;
          sstream << worker_thread->resource_name << ", " << worker_thread->task->task_name << ", ";
          for (auto idx = 0; idx < cedr_config->getPAPICounters().size(); idx++) {
            sstream << papiValues[idx];
            if (idx != cedr_config->getPAPICounters().size() - 1) {
              sstream << ", ";
            }
          }
          LOG_INFO_(PerfMon::LoggerId) << sstream.str();
          LOG_DEBUG << "Stopped measurement of PAPI counters";
        }
      }
#endif
      LOG_VERBOSE << "Successfully executed " << worker_thread->task->task_name;
      pthread_mutex_lock(thread_lock);
      worker_thread->completed_task_dequeue.push_back(task);
      worker_thread->resource_state = 0;
      pthread_mutex_unlock(thread_lock);
    } else {
      if (worker_thread->resource_state == 3) {
        pthread_mutex_unlock(thread_lock);
        break;
      }
      pthread_mutex_unlock(thread_lock);
      last_avail = 0;
      last_busy = 0;
    }
    sched_yield();
  }
  return nullptr;
}

void spawn_worker_thread(ConfigManager &cedr_config, pthread_t *pthread_handles, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                         uint8_t res_type, uint32_t global_idx, uint32_t cluster_idx) {
  const unsigned int processor_count = std::thread::hardware_concurrency();
  int ret;

  // TODO: I think we can get away with allocating the "pre-spawn"-related attributes on the stack
  // but I think we might run into issues if we try to put the pthread_arg there since the worker thread really _needs_ that throughout its execution
  auto *thread_argument = (pthread_arg *) calloc(1, sizeof(pthread_arg));  
  
  // Setup parameters for the thread
  hardware_thread_handle[global_idx].task = nullptr;
  hardware_thread_handle[global_idx].resource_state = 0;

  hardware_thread_handle[global_idx].resource_name = resource_type_names[res_type] + std::to_string(cluster_idx + 1);
  hardware_thread_handle[global_idx].thread_resource_type = (resource_type) res_type;
  hardware_thread_handle[global_idx].resource_cluster_idx = cluster_idx;
  hardware_thread_handle[global_idx].thread_avail_time = 0; // TODO: should it be set to current time instead?

  hardware_thread_handle[global_idx].cedr_config = &cedr_config;

  pthread_mutex_init(&(resource_mutex[global_idx]), nullptr);
  thread_argument->thread = &(hardware_thread_handle[global_idx]);
  thread_argument->thread_lock = &(resource_mutex[global_idx]);

  // Spawn thread, check errors, and set affinity
  ret = pthread_create(&(pthread_handles[global_idx]), nullptr, hardware_thread, (void *) thread_argument);
  if (ret != 0) {
    std::string errMsg;
    if (ret == EAGAIN) {
      errMsg = "(EAGAIN) Insufficient resources to create another thread or a "
                "system-imposed limit on the number of threads was encountered";
    } else if (ret == EINVAL) {
      errMsg = "(EINVAL) Invalid settings in attr";
    } else if (ret == EPERM) {
      errMsg = "(EPERM) No permission to set the scheduling policy and "
                "parameters specified in attr";
    }
    LOG_FATAL << "Worker thread creation failed for thread " << global_idx+1 << ": " << errMsg;
    exit(1);
  }

  // It's only worth customizing worker thread affinities if we have more than one processor to move them between!
  if (processor_count > 1) {
    // Core 0: reserved for CEDR
    // Cores 1...N: used for worker threads
    // This is the same as looping over indices 0...(N-1) and then adding 1
    int cpu_idx = 1 + (global_idx % (processor_count-1));
    LOG_DEBUG << "Setting affinity such that thread " << global_idx + 1 << " runs on CPU " << cpu_idx;
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu_idx, &cpu_set);
    pthread_setaffinity_np(pthread_handles[global_idx], sizeof(cpu_set_t), &cpu_set);
  }
  
  // If we're "loosening" thread permissions, then leave it as the default Linux CFS
  // Alternatively, if we only have one processor, setting the worker thread to a real time scheduler will ensure that 
  // the parent thread is never scheduled and the overall CEDR process hangs (unless CEDR is also SCHED_RR)
  if (!cedr_config.getLoosenThreadPermissions() && processor_count > 1) {
    struct sched_param sp;
    // Otherwise, use the "real-time" SCHED_RR with a priority of 99
    sp.sched_priority = 99;
    LOG_DEBUG << "Modifying the scheduling policy for thread " << global_idx + 1 << " to run with SCHED_RR";
    ret = pthread_setschedparam(pthread_handles[global_idx], SCHED_RR, &sp);

    if (ret != 0) {
      std::string errMsg;
      if (ret == ESRCH) {
        errMsg = "No thread with the ID thread could be found.";
      } else if (ret == EINVAL) {
        errMsg = "Requested policy is not a recognized policy, or param does not make sense for the policy.";
      } else if (ret == EPERM) {
        errMsg = "The caller does not have appropriate privileges to set the specified scheduling policy and parameters.";
      }
      LOG_FATAL << "Unable to set pthread scheduling policy for an accelerator worker thread: " << errMsg;
      exit(1);
    }
  } else {
    
  }

  LOG_DEBUG << "Finished spawning thread " << global_idx + 1;
}

void initializeThreads(ConfigManager &cedr_config, pthread_t *resource_handle, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex) { 
  // Start by configuring the affinity and scheduler of the main CEDR thread (this thread)
  pthread_t current_thread = pthread_self();

  // We only want CEDR to run on CPU 0
  cpu_set_t scheduler_affinity;
  CPU_ZERO(&scheduler_affinity);
  CPU_SET(0, &scheduler_affinity);
  pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &scheduler_affinity);

  // Sometimes, we like to override the scheduling policy followed by the linux kernel to SCHED_RR for the CEDR thread
  // If that is the case, uncomment this
  // struct sched_param main_thread;
  // if (!cedr_config.getLoosenThreadPermissions()) {
  //   main_thread.sched_priority = 99;                               // If using SCHED_RR, set this
  //   pthread_setschedparam(current_thread, SCHED_RR, &main_thread); // SCHED_RR
  // }
  
  uint32_t global_idx = 0;
  const unsigned int processor_count = std::thread::hardware_concurrency();

  for (uint8_t res_type = 0; res_type < (uint8_t) resource_type::NUM_RESOURCE_TYPES; res_type++) {
    const unsigned int workers_to_spawn = cedr_config.getResourceArray()[res_type];
    if (processor_count < workers_to_spawn) {
      LOG_WARNING << "More " << resource_type_names[res_type] << " threads are requested (" << workers_to_spawn 
                  << ") than there are available CPU cores for (" << processor_count << ") -- performance may be degraded!";
    }
    
    for (uint32_t cluster_idx = 0; cluster_idx < workers_to_spawn; cluster_idx++) {
      LOG_DEBUG << "Spawning thread " << global_idx+1 << ". This is a worker thread for resource: " << resource_type_names[res_type] << " " << cluster_idx+1;
      spawn_worker_thread(cedr_config, resource_handle, hardware_thread_handle, resource_mutex, res_type, global_idx, cluster_idx);
      global_idx++;
    }
  }
}

void cleanupThreads(ConfigManager &cedr_config) { LOG_WARNING << "CleanupHardware not implemented"; }
