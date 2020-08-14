#include "hardware.hpp"
#include <sched.h>
#include <plog/Log.h>
#include <string>

#define ON_CPU 0
#define ON_ACC 1

const variable dummy_var{.heap_ptr = nullptr};

void *hardware_thread(void *ptr) {
  pthread_t self = pthread_self();
  clockid_t clock_id;
  if (pthread_getcpuclockid(self, &clock_id) != 0) {
    LOG_FATAL << "Not able to get CLOCK ID";
    exit(1);
  };
  auto *thread_arg = (pthread_arg *)ptr;
  const variable *run_args[MAX_ARGS];
  long long expected_finish_time;
  uint8_t resource_slx;
  int cpu_name = sched_getcpu();
  LOG_DEBUG << "Starting thread " << self << " as resource name " << std::string(thread_arg->task->resource_name)
            << " and type " << std::string(thread_arg->task->resource_type) << " on cpu id " << cpu_name;

  while (true) {
    pthread_mutex_lock(thread_arg->thread_lock);
    if ((thread_arg->task->todo_task_dequeue.size() > 0)) {
      thread_arg->task->task = thread_arg->task->todo_task_dequeue.front();
      thread_arg->task->todo_task_dequeue.pop_front();
      thread_arg->task->resource_stat = 1;
      const std::vector<variable *> &args = thread_arg->task->task->args;
      void *task_run_func = thread_arg->task->task->actual_run_func;
      pthread_mutex_unlock(thread_arg->thread_lock);

      void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
      *reinterpret_cast<void **>(&run_func) = task_run_func;

      if (args.size() > MAX_ARGS) {
        LOG_ERROR
            << "Task " << std::string(thread_arg->task->task->task_name) << " has too many arguments (" << args.size()
            << ") in its dispatch call. "
            << "Please increase the maximum number of arguments supported by the framework and try running again.";
      } else {
        for (unsigned int idx = 0; idx < MAX_ARGS; idx++) {
          if (idx < args.size()) {
            run_args[idx] = args.at(idx);
          } else {
            run_args[idx] = &dummy_var;
          }
        }

        LOG_VERBOSE << "About to dispatch " << std::string(thread_arg->task->task->task_name);
        clock_gettime(CLOCK_MONOTONIC_RAW, &(thread_arg->task->task->start));
        // TODO: Cleaning this up might be nice (and enabling a scheduler that uses it)
        //(strcmp(thread_arg->task->resource_type,"cpu") == 0 )? (resource_slx = ON_CPU):(resource_slx = ON_ACC);
        // expected_finish_time = (long long)thread_arg->task->task->estimated_execution[resource_slx]*USEC2NANOSEC +
        //(long long)thread_arg->task->task->start.tv_sec*SEC2NANOSEC + (long
        // long)thread_arg->task->task->start.tv_nsec; thread_arg->task->resource_avail_time = expected_finish_time;
        run_func(run_args[0]->heap_ptr, run_args[1]->heap_ptr, run_args[2]->heap_ptr, run_args[3]->heap_ptr,
                 run_args[4]->heap_ptr, run_args[5]->heap_ptr, run_args[6]->heap_ptr, run_args[7]->heap_ptr,
                 run_args[8]->heap_ptr, run_args[9]->heap_ptr, run_args[10]->heap_ptr, run_args[11]->heap_ptr,
                 run_args[12]->heap_ptr, run_args[13]->heap_ptr, run_args[14]->heap_ptr, run_args[15]->heap_ptr,
                 run_args[16]->heap_ptr, run_args[17]->heap_ptr, run_args[18]->heap_ptr, run_args[19]->heap_ptr,
                 run_args[20]->heap_ptr, run_args[21]->heap_ptr, run_args[22]->heap_ptr, run_args[23]->heap_ptr,
                 run_args[24]->heap_ptr, run_args[25]->heap_ptr, run_args[26]->heap_ptr, run_args[27]->heap_ptr,
                 run_args[28]->heap_ptr, run_args[29]->heap_ptr, run_args[30]->heap_ptr, run_args[31]->heap_ptr,
                 run_args[32]->heap_ptr, run_args[33]->heap_ptr, run_args[34]->heap_ptr, run_args[35]->heap_ptr,
                 run_args[36]->heap_ptr, run_args[37]->heap_ptr, run_args[38]->heap_ptr, run_args[39]->heap_ptr,
                 run_args[40]->heap_ptr, run_args[41]->heap_ptr, run_args[42]->heap_ptr, run_args[43]->heap_ptr,
                 run_args[44]->heap_ptr, run_args[45]->heap_ptr, run_args[46]->heap_ptr, run_args[47]->heap_ptr,
                 run_args[48]->heap_ptr, run_args[49]->heap_ptr, run_args[50]->heap_ptr, run_args[51]->heap_ptr,
                 run_args[52]->heap_ptr, run_args[53]->heap_ptr, run_args[54]->heap_ptr, run_args[55]->heap_ptr,
                 run_args[56]->heap_ptr, run_args[57]->heap_ptr, run_args[58]->heap_ptr);
        clock_gettime(CLOCK_MONOTONIC_RAW, &(thread_arg->task->task->end));
      }
      LOG_VERBOSE << "Successfully executed " << std::string(thread_arg->task->task->task_name);

      pthread_mutex_lock((thread_arg->thread_lock));
      thread_arg->task->completed_task_dequeue.push_back(thread_arg->task->task);
      thread_arg->task->resource_stat = 0;
      pthread_mutex_unlock((thread_arg->thread_lock));
    } else {
      if (thread_arg->task->resource_stat == 3) {
        pthread_mutex_unlock(thread_arg->thread_lock);
        break;
      }
      pthread_mutex_unlock(thread_arg->thread_lock);
    }
    pthread_yield();
  }
  return nullptr;
}

void initializeCPUs(pthread_t *resource_handle, running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex) {

  const unsigned int resource_count = CORE_RESOURCE_COUNT;

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
#ifdef DAEMON
    CPU_SET(i + 1, &(resource_affinity[i]));
#else
    CPU_SET(i + 1, &(resource_affinity[i]));
#endif
    pthread_attr_setaffinity_np(&(resource_attr[i]), sizeof(cpu_set_t), &(resource_affinity[i]));
    pthread_attr_setinheritsched(&(resource_attr[i]), PTHREAD_EXPLICIT_SCHED);
    int ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_RR); // SCHED_OTHER
    if (ret != 0) {
      LOG_FATAL << "Unable to set CPU pthread scheduling policy";
      exit(1);
    }
    p1[i].sched_priority = 99;                         // COMMENT OUT
    pthread_attr_setschedparam(&resource_attr[i], p1); // COMMENT OUT
  }

  auto *thread_argument = (pthread_arg *)malloc(resource_count * sizeof(pthread_arg));

  for (int i = 0; i < CORE_RESOURCE_COUNT; i++) {
    LOG_VERBOSE << "Spawning hardware thread for resource " << i;
    hardware_thread_handle[i].task = nullptr;
    hardware_thread_handle[i].resource_stat = 0;
    strncpy(hardware_thread_handle[i].resource_name, std::string("Core " + std::to_string(i + 1)).c_str(),
            sizeof(hardware_thread_handle[i].resource_name) - 1);
    strncpy(hardware_thread_handle[i].resource_type, "cpu", sizeof(hardware_thread_handle[i].resource_type) - 1);

    hardware_thread_handle[i].resource_config_input = i;
    pthread_mutex_init(&(resource_mutex[i]), nullptr);
    thread_argument[i].task = &(hardware_thread_handle[i]);
    thread_argument[i].thread_lock = &(resource_mutex[i]);

    int thread_check =
        pthread_create(&(resource_handle[i]), &(resource_attr[i]), hardware_thread, (void *)&(thread_argument[i]));
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

#ifdef ARM
void initializeFFTs(pthread_t *resource_handle, running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex) {

  const unsigned int resource_count = FFT_RESOURCE_COUNT;

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
    CPU_SET(i + 1, &(resource_affinity[i]));
    pthread_attr_setaffinity_np(&(resource_attr[i]), sizeof(cpu_set_t), &(resource_affinity[i]));
    pthread_attr_setinheritsched(&(resource_attr[i]), PTHREAD_EXPLICIT_SCHED);
    int ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_RR); // SCHED_OTHERS
    if (ret != 0) {
      LOG_FATAL << "Unable to set accelerator pthread scheduling policy";
      exit(1);
    }
    p1[i].sched_priority = 99;                               // COMMENT OUT
    pthread_attr_setschedparam(&(resource_attr[i]), &p1[i]); // COMMENT OUT
  }

  for (int i = 0; i < resource_count; i++) {
    const int i_offset = CORE_RESOURCE_COUNT + i;
    hardware_thread_handle[i_offset].task = nullptr;
    hardware_thread_handle[i_offset].resource_stat = 0;
    strncpy(hardware_thread_handle[i_offset].resource_name, std::string("FFT " + std::to_string(i + 1)).c_str(),
            sizeof(hardware_thread_handle[i_offset].resource_name) - 1);
    strncpy(hardware_thread_handle[i_offset].resource_type, "fft",
            sizeof(hardware_thread_handle[i_offset].resource_type) - 1);

    hardware_thread_handle[i_offset].resource_config_input = i;
    pthread_mutex_init(&(resource_mutex[i_offset]), nullptr);
    thread_argument[i].task = &(hardware_thread_handle[i_offset]);
    thread_argument[i].thread_lock = &(resource_mutex[i_offset]);

    int thread_check = pthread_create(&(resource_handle[i_offset]), &(resource_attr[i]), hardware_thread,
                                      (void *)&(thread_argument[i]));
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

void initializeMMULTs(pthread_t *resource_handle, running_task *hardware_thread_handle,
                      pthread_mutex_t *resource_mutex) {
  const unsigned int resource_count = MMULT_RESOURCE_COUNT;

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
    CPU_SET(i + 2, &(resource_affinity[i]));
    pthread_attr_setaffinity_np(&(resource_attr[i]), sizeof(cpu_set_t), &(resource_affinity[i]));
    pthread_attr_setinheritsched(&(resource_attr[i]), PTHREAD_EXPLICIT_SCHED);
    int ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_RR); // SCHED_OTHERS
    if (ret != 0) {
      LOG_FATAL << "Unable to set accelerator pthread scheduling policy";
      exit(1);
    }
    p1[i].sched_priority = 99;                               // COMMENT OUT
    pthread_attr_setschedparam(&(resource_attr[i]), &p1[i]); // COMMENT OUT
  }

  for (int i = 0; i < resource_count; i++) {
    const int i_offset = CORE_RESOURCE_COUNT + FFT_RESOURCE_COUNT + i;
    hardware_thread_handle[i_offset].task = nullptr;
    hardware_thread_handle[i_offset].resource_stat = 0;
    strncpy(hardware_thread_handle[i_offset].resource_name, std::string("MMULT " + std::to_string(i + 1)).c_str(),
            sizeof(hardware_thread_handle[i_offset].resource_name) - 1);
    strncpy(hardware_thread_handle[i_offset].resource_type, "mmult",
            sizeof(hardware_thread_handle[i_offset].resource_type) - 1);

    hardware_thread_handle[i_offset].resource_config_input = i;
    pthread_mutex_init(&(resource_mutex[i_offset]), nullptr);
    thread_argument[i].task = &(hardware_thread_handle[i_offset]);
    thread_argument[i].thread_lock = &(resource_mutex[i_offset]);

    int thread_check = pthread_create(&(resource_handle[i_offset]), &(resource_attr[i]), hardware_thread,
                                      (void *)&(thread_argument[i]));
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
#endif

void initializeHardware(pthread_t *resource_handle, running_task *hardware_thread_handle,
                        pthread_mutex_t *resource_mutex) {
  LOG_DEBUG << "Performing CPU thread initialization";
  initializeCPUs(resource_handle, hardware_thread_handle, resource_mutex);
#ifdef ARM
  LOG_DEBUG << "Performing FFT initialization";
  initializeFFTs(resource_handle, hardware_thread_handle, resource_mutex);
  LOG_DEBUG << "Performing MMULT initialization";
  initializeMMULTs(resource_handle, hardware_thread_handle, resource_mutex);
#else
  LOG_DEBUG << "Skipping accelerator initialization for non-ARM build";
#endif
  LOG_DEBUG << "Hardware initialization complete";
}

void cleanupHardware() { LOG_WARNING << "cleanupHardware not implemented"; }
