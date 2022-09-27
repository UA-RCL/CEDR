#include "threads.hpp"
#include <dlfcn.h>
#include <sched.h>
#include <thread>

#if defined(USEPAPI)
#include "performance_monitor.hpp"
#include <papi/papi.h>
#endif

#include <plog/Log.h>
#include <string>

#define ON_CPU 0
#define ON_ACC 1

const variable dummy_var{.name="", .num_bytes=0, .heap_ptr = nullptr}; // Out of order initialization doesn't work with g++ (works with clang)

void *hardware_thread(void *ptr) {
  pthread_t self = pthread_self();
  clockid_t clock_id;
  if (pthread_getcpuclockid(self, &clock_id) != 0) {
    LOG_FATAL << "Not able to get CLOCK ID";
    exit(1);
  }
  auto *thread_arg = (pthread_arg *)ptr;
  // Since we spend a lot of time dereferencing these pointers from thread_arg, might as well do it once
  auto *worker_thread = thread_arg->thread;
  auto *thread_lock = thread_arg->thread_lock;
  auto *cedr_config = worker_thread->cedr_config;

  const variable *run_args[MAX_ARGS];
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
  } else {
    LOG_DEBUG << "PAPI is not initialized!";
  }
#endif

  // std::cout << "Entering hardware thread while loop " << std::endl;
  while (true) {
    pthread_mutex_lock(thread_lock);
    if ((!worker_thread->todo_task_dequeue.empty())) {
      auto *task = worker_thread->todo_task_dequeue.front();
      worker_thread->todo_task_dequeue.pop_front();
      worker_thread->task = task;
      worker_thread->resource_state = 1;
      const std::vector<variable *> &args = task->args;
      void *task_run_func = task->actual_run_func;
      pthread_mutex_unlock(thread_lock);

      void *dlhandle = task->app_pnt->dlhandle;
      unsigned int *node_idx_ptr = (unsigned int *) dlsym(dlhandle, "__CEDR_TASK_ID__");
      if (dlerror() == nullptr) {
        LOG_DEBUG << "Changing CEDR Task ID in thread " << cpu_name << " from " << *node_idx_ptr << " to " << task->task_id;
        *node_idx_ptr = task->task_id;
      }
      if (worker_thread->thread_resource_type != resource_type::cpu) {
        unsigned int *cluster_idx_ptr = (unsigned int *) dlsym(dlhandle, "__CEDR_CLUSTER_IDX__");
        if (dlerror() == nullptr) {
          LOG_DEBUG << "Changing CEDR cluster index in thread " << cpu_name << " from " << *cluster_idx_ptr << " to " << worker_thread->resource_cluster_idx;
          *cluster_idx_ptr = worker_thread->resource_cluster_idx;
        }
      }

      void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *, void *);
      *reinterpret_cast<void **>(&run_func) = task_run_func;

      if (args.size() > MAX_ARGS) {
        LOG_ERROR << "Task " << worker_thread->task->task_name << " has too many arguments (" << args.size() << ") in its dispatch call. "
                  << "Please increase the maximum number of arguments supported by the framework and try running again.";
      } else {
        for (unsigned int idx = 0; idx < MAX_ARGS; idx++) {
          if (idx < args.size()) {
            run_args[idx] = args.at(idx);
          } else {
            run_args[idx] = &dummy_var;
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
          worker_thread->task->prev_idle_time = (last_busy - last_avail);
        } else {
          worker_thread->task->prev_idle_time = 0;
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

        run_func(run_args[0]->heap_ptr, run_args[1]->heap_ptr, run_args[2]->heap_ptr, run_args[3]->heap_ptr, run_args[4]->heap_ptr, run_args[5]->heap_ptr, run_args[6]->heap_ptr,
                 run_args[7]->heap_ptr, run_args[8]->heap_ptr, run_args[9]->heap_ptr, run_args[10]->heap_ptr, run_args[11]->heap_ptr, run_args[12]->heap_ptr,
                 run_args[13]->heap_ptr, run_args[14]->heap_ptr, run_args[15]->heap_ptr, run_args[16]->heap_ptr, run_args[17]->heap_ptr, run_args[18]->heap_ptr,
                 run_args[19]->heap_ptr, run_args[20]->heap_ptr, run_args[21]->heap_ptr, run_args[22]->heap_ptr, run_args[23]->heap_ptr, run_args[24]->heap_ptr,
                 run_args[25]->heap_ptr, run_args[26]->heap_ptr, run_args[27]->heap_ptr, run_args[28]->heap_ptr, run_args[29]->heap_ptr, run_args[30]->heap_ptr,
                 run_args[31]->heap_ptr, run_args[32]->heap_ptr, run_args[33]->heap_ptr, run_args[34]->heap_ptr, run_args[35]->heap_ptr, run_args[36]->heap_ptr,
                 run_args[37]->heap_ptr, run_args[38]->heap_ptr, run_args[39]->heap_ptr, run_args[40]->heap_ptr, run_args[41]->heap_ptr, run_args[42]->heap_ptr,
                 run_args[43]->heap_ptr, run_args[44]->heap_ptr, run_args[45]->heap_ptr, run_args[46]->heap_ptr, run_args[47]->heap_ptr, run_args[48]->heap_ptr,
                 run_args[49]->heap_ptr, run_args[50]->heap_ptr, run_args[51]->heap_ptr, run_args[52]->heap_ptr, run_args[53]->heap_ptr, run_args[54]->heap_ptr,
                 run_args[55]->heap_ptr, run_args[56]->heap_ptr, run_args[57]->heap_ptr, run_args[58]->heap_ptr, run_args[59]->heap_ptr, run_args[60]->heap_ptr,
                 run_args[61]->heap_ptr, run_args[62]->heap_ptr, run_args[63]->heap_ptr, run_args[64]->heap_ptr, run_args[65]->heap_ptr, run_args[66]->heap_ptr,
                 run_args[67]->heap_ptr, run_args[68]->heap_ptr, run_args[69]->heap_ptr, run_args[70]->heap_ptr, run_args[71]->heap_ptr, run_args[72]->heap_ptr,
                 run_args[73]->heap_ptr, run_args[74]->heap_ptr, run_args[75]->heap_ptr, run_args[76]->heap_ptr, run_args[77]->heap_ptr, run_args[78]->heap_ptr);
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
    pthread_yield();
  }
  return nullptr;
}

void initializeCPUs(ConfigManager &cedr_config, pthread_t *resource_handle, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex) {

  const unsigned int resource_count = cedr_config.getResourceArray()[(uint8_t) resource_type::cpu];
  const unsigned int processor_count = std::thread::hardware_concurrency();

  if (processor_count < resource_count) {
    LOG_WARNING << "More CPU threads are requested (" << resource_count << ") than there are available CPU cores for (" << processor_count << ") -- performance may be degraded!";
  }

  auto *resource_attr = (pthread_attr_t *)malloc(resource_count * sizeof(pthread_attr_t));
  auto *resource_affinity = (cpu_set_t *)malloc(resource_count * sizeof(cpu_set_t));

  cpu_set_t scheduler_affinity;
  CPU_ZERO(&scheduler_affinity);
  CPU_SET(0, &scheduler_affinity);

  pthread_t current_thread = pthread_self();
  pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &scheduler_affinity);
  // struct sched_param main_thread;
  // main_thread.sched_priority = 99;  // If using SCHED_RR, set this
  // pthread_setschedparam(current_thread, SCHED_OTHER, &main_thread); // SCHED_RR
  auto *p1 = (struct sched_param *)malloc(resource_count * sizeof(struct sched_param));

  for (int i = 0; i < resource_count; i++) {
    LOG_VERBOSE << "Setting resource affinity and scheduler policy for resource " << i;
    pthread_attr_init(&(resource_attr[i]));
    CPU_ZERO(&(resource_affinity[i]));
    CPU_SET((i + 1) % processor_count, &(resource_affinity[i]));
    pthread_attr_setaffinity_np(&(resource_attr[i]), sizeof(cpu_set_t), &(resource_affinity[i]));
    pthread_attr_setinheritsched(&(resource_attr[i]), PTHREAD_EXPLICIT_SCHED);
    int ret;
    if (cedr_config.getLoosenThreadPermissions()) {
      ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_OTHER);
    } else {
      ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_RR);
    }
    if (ret != 0) {
      LOG_FATAL << "Unable to set CPU pthread scheduling policy";
      exit(1);
    }
    if (!cedr_config.getLoosenThreadPermissions()) {
      p1[i].sched_priority = 99;
      pthread_attr_setschedparam(&resource_attr[i], p1);
    }
  }

  auto *thread_argument = (pthread_arg *)malloc(resource_count * sizeof(pthread_arg));

  for (int i = 0; i < cedr_config.getResourceArray()[(uint8_t) resource_type::cpu]; i++) {
    LOG_VERBOSE << "Spawning hardware thread for resource " << i;
    hardware_thread_handle[i].task = nullptr;
    hardware_thread_handle[i].resource_state = 0;

    hardware_thread_handle[i].resource_name = "Core " + std::to_string(i + 1);
    hardware_thread_handle[i].thread_resource_type = resource_type::cpu;
    hardware_thread_handle[i].resource_cluster_idx = i;
    hardware_thread_handle[i].thread_avail_time = 0; // TODO: Should it be set to current time instead?

    hardware_thread_handle[i].cedr_config = &cedr_config;

    pthread_mutex_init(&(resource_mutex[i]), nullptr);
    thread_argument[i].thread = &(hardware_thread_handle[i]);
    thread_argument[i].thread_lock = &(resource_mutex[i]);

    int thread_check = pthread_create(&(resource_handle[i]), &(resource_attr[i]), hardware_thread, (void *)&(thread_argument[i]));
    if (thread_check != 0) {
      std::string errMsg;
      if (thread_check == EAGAIN) {
        errMsg = "(EAGAIN) Insufficient resources to create another thread or a "
                 "system-imposed limit on the number of threads was encountered";
      } else if (thread_check == EINVAL) {
        errMsg = "(EINVAL) Invalid settings in attr";
      } else if (thread_check == EPERM) {
        errMsg = "(EPERM) No permission to set the scheduling policy and "
                 "parameters specified in attr";
      }
      LOG_FATAL << "CPU thread creation failed for resource index " << i + 1 << ": " << errMsg;
      exit(1);
    }
  }
}

void initializeFFTs(ConfigManager &cedr_config, pthread_t *resource_handle, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex) {

  const unsigned int resource_count = cedr_config.getResourceArray()[(uint8_t) resource_type::fft];
  const unsigned int processor_count = std::thread::hardware_concurrency();

  if (processor_count < resource_count) {
    LOG_WARNING << "More FFT threads are requested (" << resource_count << ") than there are available CPU cores for (" << processor_count << ") -- performance may be degraded!";
  }

  if (resource_count == 0) {
    LOG_DEBUG << "Skipping FFT initialization because none are requested";
    return;
  }

  // Note: Anything allocated _here_ should use the standard i=0 to FFT_RESOURCE_COUNT index expressions
  // Anything _passed in_ should use the i_offset indexing expressions
  auto *resource_attr = (pthread_attr_t *)calloc(resource_count, sizeof(pthread_attr_t));
  auto *resource_affinity = (cpu_set_t *)calloc(resource_count, sizeof(cpu_set_t));
  auto *p1 = (struct sched_param *)calloc(resource_count, sizeof(struct sched_param));
  auto *thread_argument = (pthread_arg *)calloc(resource_count, sizeof(pthread_arg));

  for (int i = 0; i < resource_count; i++) {
    LOG_DEBUG << "Initializing FFT " << i + 1;
    pthread_attr_init(&(resource_attr[i]));
    CPU_ZERO(&(resource_affinity[i]));
    CPU_SET((processor_count - i - 2) % processor_count, &(resource_affinity[i]));
    pthread_attr_setaffinity_np(&(resource_attr[i]), sizeof(cpu_set_t), &(resource_affinity[i]));
    pthread_attr_setinheritsched(&(resource_attr[i]), PTHREAD_EXPLICIT_SCHED);
    int ret;
    if (cedr_config.getLoosenThreadPermissions()) {
      ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_OTHER);
    } else {
      ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_RR);
    }
    if (ret != 0) {
      LOG_FATAL << "Unable to set accelerator pthread scheduling policy";
      exit(1);
    }
    if (!cedr_config.getLoosenThreadPermissions()) {
      p1[i].sched_priority = 99;
      pthread_attr_setschedparam(&(resource_attr[i]), &p1[i]);
    }
  }

  for (int i = 0; i < resource_count; i++) {
    const unsigned int i_offset = cedr_config.getResourceArray()[(uint8_t) resource_type::cpu] + i;
    hardware_thread_handle[i_offset].task = nullptr;
    hardware_thread_handle[i_offset].resource_state = 0;

    hardware_thread_handle[i_offset].resource_name = "FFT " + std::to_string(i + 1);
    hardware_thread_handle[i_offset].thread_resource_type = resource_type::fft;
    hardware_thread_handle[i_offset].resource_cluster_idx = i;
    hardware_thread_handle[i_offset].thread_avail_time = 0; // TODO: Should it be set to current time instead?

    hardware_thread_handle[i_offset].cedr_config = &cedr_config;

    pthread_mutex_init(&(resource_mutex[i_offset]), nullptr);
    thread_argument[i].thread = &(hardware_thread_handle[i_offset]);
    thread_argument[i].thread_lock = &(resource_mutex[i_offset]);

    int thread_check = pthread_create(&(resource_handle[i_offset]), &(resource_attr[i]), hardware_thread, (void *)&(thread_argument[i]));
    if (thread_check != 0) {
      std::string errMsg;
      if (thread_check == EAGAIN) {
        errMsg = "(EAGAIN) Insufficient resources to create another thread or a "
                 "system-imposed limit on the number of threads was encountered";
      } else if (thread_check == EINVAL) {
        errMsg = "(EINVAL) Invalid settings in attr";
      } else if (thread_check == EPERM) {
        errMsg = "(EPERM) No permission to set the scheduling policy and "
                 "parameters specified in attr";
      }
      LOG_FATAL << "FFT thread creation failed for resource index " << i + 1 << ": " << errMsg;
      exit(1);
    }
  }

  LOG_DEBUG << "Finished FFT initialization";
}

void initializeMMULTs(ConfigManager &cedr_config, pthread_t *resource_handle, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex) {
  const unsigned int resource_count = cedr_config.getResourceArray()[(uint8_t) resource_type::mmult];
  const unsigned int processor_count = std::thread::hardware_concurrency();

  if (processor_count < resource_count) {
    LOG_WARNING << "More MMULT threads are requested (" << resource_count << ") than there are available CPU cores for (" << processor_count << ") -- performance may be degraded!";
  }

  if (resource_count == 0) {
    LOG_DEBUG << "Skipping MMULT initialization because none are requested";
    return;
  }

  // Note: Anything allocated _here_ should use the standard i=0 to MMULT_RESOURCE_COUNT index expressions
  // Anything _passed in_ should use the i_offset indexing expressions
  auto *resource_attr = (pthread_attr_t *)calloc(resource_count, sizeof(pthread_attr_t));
  auto *resource_affinity = (cpu_set_t *)calloc(resource_count, sizeof(cpu_set_t));
  auto *p1 = (struct sched_param *)calloc(resource_count, sizeof(struct sched_param));
  auto *thread_argument = (pthread_arg *)calloc(resource_count, sizeof(pthread_arg));

  for (int i = 0; i < resource_count; i++) {
    LOG_DEBUG << "Initializing MMULT " << i + 1;
    pthread_attr_init(&(resource_attr[i]));
    CPU_ZERO(&(resource_affinity[i]));
    CPU_SET((processor_count - i - 1) % processor_count, &(resource_affinity[i]));
    pthread_attr_setaffinity_np(&(resource_attr[i]), sizeof(cpu_set_t), &(resource_affinity[i]));
    pthread_attr_setinheritsched(&(resource_attr[i]), PTHREAD_EXPLICIT_SCHED);
    int ret;
    if (cedr_config.getLoosenThreadPermissions()) {
      ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_OTHER);
    } else {
      ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_RR);
    }
    if (ret != 0) {
      LOG_FATAL << "Unable to set accelerator pthread scheduling policy";
      exit(1);
    }
    if (!cedr_config.getLoosenThreadPermissions()) {
      p1[i].sched_priority = 99;
      pthread_attr_setschedparam(&(resource_attr[i]), &p1[i]);
    }
  }

  for (int i = 0; i < resource_count; i++) {
    const unsigned int i_offset = cedr_config.getResourceArray()[(uint8_t) resource_type::cpu] + cedr_config.getResourceArray()[(uint8_t) resource_type::fft] + i;
    hardware_thread_handle[i_offset].task = nullptr;
    hardware_thread_handle[i_offset].resource_state = 0;

    hardware_thread_handle[i_offset].resource_name = "MMULT " + std::to_string(i + 1);
    hardware_thread_handle[i_offset].thread_resource_type = resource_type::mmult;
    hardware_thread_handle[i_offset].resource_cluster_idx = i;
    hardware_thread_handle[i_offset].thread_avail_time = 0; // TODO: Should it be set to current time instead?

    hardware_thread_handle[i_offset].cedr_config = &cedr_config;

    pthread_mutex_init(&(resource_mutex[i_offset]), nullptr);
    thread_argument[i].thread = &(hardware_thread_handle[i_offset]);
    thread_argument[i].thread_lock = &(resource_mutex[i_offset]);

    int thread_check = pthread_create(&(resource_handle[i_offset]), &(resource_attr[i]), hardware_thread, (void *)&(thread_argument[i]));
    if (thread_check != 0) {
      std::string errMsg;
      if (thread_check == EAGAIN) {
        errMsg = "(EAGAIN) Insufficient resources to create another thread or a "
                 "system-imposed limit on the number of threads was encountered";
      } else if (thread_check == EINVAL) {
        errMsg = "(EINVAL) Invalid settings in attr";
      } else if (thread_check == EPERM) {
        errMsg = "(EPERM) No permission to set the scheduling policy and "
                 "parameters specified in attr";
      }
      LOG_FATAL << "MMULT thread creation failed for resource index " << i + 1 << ": " << errMsg;
      exit(1);
    }
  }

  LOG_DEBUG << "Finished MMULT initialization";
}

void initializeGPUs(ConfigManager &cedr_config, pthread_t *resource_handle, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex) {
  const unsigned int resource_count = cedr_config.getResourceArray()[(uint8_t) resource_type::gpu];
  const unsigned int processor_count = std::thread::hardware_concurrency();

  if (processor_count < resource_count) {
    LOG_WARNING << "More GPU threads are requested (" << resource_count << ") than there are available CPU cores for (" << processor_count << ") -- performance may be degraded!";
  }

  if (resource_count == 0) {
    LOG_DEBUG << "Skipping GPU initialization because none are requested";
    return;
  }

  // Note: Anything allocated _here_ should use the standard i=0 to GPU_RESOURCE_COUNT index expressions
  // Anything _passed in_ should use the i_offset indexing expressions
  auto *resource_attr = (pthread_attr_t *)calloc(resource_count, sizeof(pthread_attr_t));
  auto *resource_affinity = (cpu_set_t *)calloc(resource_count, sizeof(cpu_set_t));
  auto *p1 = (struct sched_param *)calloc(resource_count, sizeof(struct sched_param));
  auto *thread_argument = (pthread_arg *)calloc(resource_count, sizeof(pthread_arg));

  for (int i = 0; i < resource_count; i++) {
    LOG_DEBUG << "Initializing GPU " << i + 1;
    pthread_attr_init(&(resource_attr[i]));
    CPU_ZERO(&(resource_affinity[i]));
    CPU_SET((processor_count - i - 1) % processor_count, &(resource_affinity[i]));
    pthread_attr_setaffinity_np(&(resource_attr[i]), sizeof(cpu_set_t), &(resource_affinity[i]));
    pthread_attr_setinheritsched(&(resource_attr[i]), PTHREAD_EXPLICIT_SCHED);
    int ret;
    if (cedr_config.getLoosenThreadPermissions()) {
      ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_OTHER);
    } else {
      ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_RR);
    }
    if (ret != 0) {
      LOG_FATAL << "Unable to set accelerator pthread scheduling policy";
      exit(1);
    }
    if (!cedr_config.getLoosenThreadPermissions()) {
      p1[i].sched_priority = 99;
      pthread_attr_setschedparam(&(resource_attr[i]), &p1[i]);
    }
  }

  for (int i = 0; i < resource_count; i++) {
    const unsigned int i_offset = cedr_config.getResourceArray()[(uint8_t) resource_type::cpu] + cedr_config.getResourceArray()[(uint8_t) resource_type::fft] + cedr_config.getResourceArray()[(uint8_t) resource_type::mmult] + i;
    hardware_thread_handle[i_offset].task = nullptr;
    hardware_thread_handle[i_offset].resource_state = 0;

    hardware_thread_handle[i_offset].resource_name = "GPU " + std::to_string(i + 1);
    hardware_thread_handle[i_offset].thread_resource_type = resource_type::gpu;
    hardware_thread_handle[i_offset].resource_cluster_idx = i;
    hardware_thread_handle[i_offset].thread_avail_time = 0; // TODO: Should it be set to current time instead?

    hardware_thread_handle[i_offset].cedr_config = &cedr_config;

    pthread_mutex_init(&(resource_mutex[i_offset]), nullptr);
    thread_argument[i].thread = &(hardware_thread_handle[i_offset]);
    thread_argument[i].thread_lock = &(resource_mutex[i_offset]);

    int thread_check = pthread_create(&(resource_handle[i_offset]), &(resource_attr[i]), hardware_thread, (void *)&(thread_argument[i]));
    if (thread_check != 0) {
      std::string errMsg;
      if (thread_check == EAGAIN) {
        errMsg = "(EAGAIN) Insufficient resources to create another thread or a "
                 "system-imposed limit on the number of threads was encountered";
      } else if (thread_check == EINVAL) {
        errMsg = "(EINVAL) Invalid settings in attr";
      } else if (thread_check == EPERM) {
        errMsg = "(EPERM) No permission to set the scheduling policy and "
                 "parameters specified in attr";
      }
      LOG_FATAL << "GPU thread creation failed for resource index " << i + 1 << ": " << errMsg;
      exit(1);
    }
  }

  LOG_DEBUG << "Finished GPU initialization";
}

void initializeThreads(ConfigManager &cedr_config, pthread_t *resource_handle, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex) {
  LOG_DEBUG << "Initializing CPU worker threads";
  initializeCPUs(cedr_config, resource_handle, hardware_thread_handle, resource_mutex);
  LOG_DEBUG << "Initializing FFT worker threads";
  initializeFFTs(cedr_config, resource_handle, hardware_thread_handle, resource_mutex);
  LOG_DEBUG << "Initializing MMULT worker threads";
  initializeMMULTs(cedr_config, resource_handle, hardware_thread_handle, resource_mutex);
  LOG_DEBUG << "Initializing GPU worker threads";
  initializeGPUs(cedr_config, resource_handle, hardware_thread_handle, resource_mutex);
  LOG_DEBUG << "Hardware initialization complete";
}

void cleanupThreads(ConfigManager &cedr_config) { LOG_WARNING << "CleanupHardware not implemented"; }
