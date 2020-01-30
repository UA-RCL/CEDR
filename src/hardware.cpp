#include "hardware.hpp"
#include <sched.h>
#include <plog/Log.h>
#include <string>

void *hardware_thread(void *ptr) {
  pthread_t self = pthread_self();
  clockid_t clock_id;
  if (pthread_getcpuclockid(self, &clock_id) != 0) {
    LOG_FATAL << "Not able to get CLOCK ID";
    exit(1);
  };
  auto *thread_arg = (pthread_arg *)ptr;

  int cpu_name = sched_getcpu();
  LOG_DEBUG << "Starting thread " << self << " as resource name " << std::string(thread_arg->task->resource_name)
            << " and type " << std::string(thread_arg->task->resource_type) << " on cpu id " << cpu_name;

  while (true) {
    pthread_mutex_lock((thread_arg->thread_lock));
    if (thread_arg->task->resource_stat == 1) {
      pthread_mutex_unlock((thread_arg->thread_lock));
      const std::vector<variable *> &args = thread_arg->task->task->args;
      void *task_run_func = thread_arg->task->task->actual_run_func;

      // TODO: https://github.com/mackncheesiest/DSSoCEmulator/issues/6
      //      void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
      //      void *,
      //                       void *, void *, void *, void *, void *, void *, void *, void *);
      //      *reinterpret_cast<void **>(&run_func) = thread_arg->task->task->actual_run_func;

      //      if (args.size() > MAX_ARGS) {
      //        LOG_ERROR
      //            << "Task " << std::string(thread_arg->task->task->task_name)
      //            << " has too many arguments in its dispatch call. "
      //            << "Please increase the maximum number of arguments supported by the framework and try running
      //            again.";
      //      } else {
      //        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
      //                 args.at(4)->heap_ptr, args.at(5)->heap_ptr, args.at(6)->heap_ptr, args.at(7)->heap_ptr,
      //                 args.at(8)->heap_ptr, args.at(9)->heap_ptr, args.at(10)->heap_ptr, args.at(11)->heap_ptr,
      //                 args.at(12)->heap_ptr, args.at(13)->heap_ptr, args.at(14)->heap_ptr, args.at(15)->heap_ptr,
      //                 args.at(16)->heap_ptr, args.at(17)->heap_ptr, args.at(18)->heap_ptr, args.at(19)->heap_ptr);
      //      }

      LOG_VERBOSE << "About to dispatch " << std::string(thread_arg->task->task->task_name);
      clock_gettime(CLOCK_MONOTONIC_RAW, &(thread_arg->task->task->start));
      if (args.empty()) {
        const void (*run_func)();
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func();
      } else if (args.size() == 1) {
        void (*run_func)(void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr);
      } else if (args.size() == 2) {
        void (*run_func)(void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr);
      } else if (args.size() == 3) {
        void (*run_func)(void *, void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr);
      } else if (args.size() == 4) {
        void (*run_func)(void *, void *, void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr);
      } else if (args.size() == 5) {
        void (*run_func)(void *, void *, void *, void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
                 args.at(4)->heap_ptr);
      } else if (args.size() == 6) {
        void (*run_func)(void *, void *, void *, void *, void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
                 args.at(4)->heap_ptr, args.at(5)->heap_ptr);
      } else if (args.size() == 7) {
        void (*run_func)(void *, void *, void *, void *, void *, void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
                 args.at(4)->heap_ptr, args.at(5)->heap_ptr, args.at(6)->heap_ptr);
      } else if (args.size() == 8) {
        void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
                 args.at(4)->heap_ptr, args.at(5)->heap_ptr, args.at(6)->heap_ptr, args.at(7)->heap_ptr);
      } else if (args.size() == 9) {
        void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
                 args.at(4)->heap_ptr, args.at(5)->heap_ptr, args.at(6)->heap_ptr, args.at(7)->heap_ptr,
                 args.at(8)->heap_ptr);
      } else if (args.size() == 10) {
        void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
                 args.at(4)->heap_ptr, args.at(5)->heap_ptr, args.at(6)->heap_ptr, args.at(7)->heap_ptr,
                 args.at(8)->heap_ptr, args.at(9)->heap_ptr);
      } else if (args.size() == 11) {
        void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
                 args.at(4)->heap_ptr, args.at(5)->heap_ptr, args.at(6)->heap_ptr, args.at(7)->heap_ptr,
                 args.at(8)->heap_ptr, args.at(9)->heap_ptr, args.at(10)->heap_ptr);
      } else if (args.size() == 12) {
        void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                         void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
                 args.at(4)->heap_ptr, args.at(5)->heap_ptr, args.at(6)->heap_ptr, args.at(7)->heap_ptr,
                 args.at(8)->heap_ptr, args.at(9)->heap_ptr, args.at(10)->heap_ptr, args.at(11)->heap_ptr);
      } else if (args.size() == 13) {
        void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                         void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
                 args.at(4)->heap_ptr, args.at(5)->heap_ptr, args.at(6)->heap_ptr, args.at(7)->heap_ptr,
                 args.at(8)->heap_ptr, args.at(9)->heap_ptr, args.at(10)->heap_ptr, args.at(11)->heap_ptr,
                 args.at(12)->heap_ptr);
      } else if (args.size() == 14) {
        void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                         void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
                 args.at(4)->heap_ptr, args.at(5)->heap_ptr, args.at(6)->heap_ptr, args.at(7)->heap_ptr,
                 args.at(8)->heap_ptr, args.at(9)->heap_ptr, args.at(10)->heap_ptr, args.at(11)->heap_ptr,
                 args.at(12)->heap_ptr, args.at(13)->heap_ptr);
      } else if (args.size() == 15) {
        void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                         void *, void *, void *);
        *reinterpret_cast<void **>(&run_func) = task_run_func;
        run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
                 args.at(4)->heap_ptr, args.at(5)->heap_ptr, args.at(6)->heap_ptr, args.at(7)->heap_ptr,
                 args.at(8)->heap_ptr, args.at(9)->heap_ptr, args.at(10)->heap_ptr, args.at(11)->heap_ptr,
                 args.at(12)->heap_ptr, args.at(13)->heap_ptr, args.at(14)->heap_ptr);
      } else {
        LOG_ERROR << "Task " << std::string(thread_arg->task->task->task_name)
                  << " has too many arguments in its dispatch call. "
                  << "Please increase the maximum number of supported arguments and try running again.";
      }
      clock_gettime(CLOCK_MONOTONIC_RAW, &(thread_arg->task->task->end));
      LOG_VERBOSE << "Successfully executed " << std::string(thread_arg->task->task->task_name);

      pthread_mutex_lock((thread_arg->thread_lock));
      thread_arg->task->resource_stat = 2;
      pthread_mutex_unlock((thread_arg->thread_lock));
    } else if (thread_arg->task->resource_stat == 3) {
      pthread_mutex_unlock((thread_arg->thread_lock));
      break;
    } else {
      pthread_mutex_unlock((thread_arg->thread_lock));
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
  auto *p1 = (struct sched_param *)malloc(resource_count * sizeof(struct sched_param));

  for (int i = 0; i < resource_count; i++) {
    pthread_attr_init(&(resource_attr[i]));
    CPU_ZERO(&(resource_affinity[i]));
    CPU_SET(i + 1, &(resource_affinity[i]));
    pthread_attr_setaffinity_np(&(resource_attr[i]), sizeof(cpu_set_t), &(resource_affinity[i]));
    pthread_attr_setinheritsched(&(resource_attr[i]), PTHREAD_EXPLICIT_SCHED);
    int ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_RR);
    if (ret != 0) {
      LOG_FATAL << "Unable to set CPU pthread scheduling policy";
      exit(1);
    }
    p1[i].sched_priority = 99;
    pthread_attr_setschedparam(&resource_attr[i], p1);
  }

  auto *thread_argument = (pthread_arg *)malloc(resource_count * sizeof(pthread_arg));

  for (int i = 0; i < CORE_RESOURCE_COUNT; i++) {
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
void initializeAccelerators(pthread_t *resource_handle, running_task *hardware_thread_handle,
                            pthread_mutex_t *resource_mutex) {

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
    CPU_SET(i + 2, &(resource_affinity[i]));
    pthread_attr_setaffinity_np(&(resource_attr[i]), sizeof(cpu_set_t), &(resource_affinity[i]));
    pthread_attr_setinheritsched(&(resource_attr[i]), PTHREAD_EXPLICIT_SCHED);
    int ret = pthread_attr_setschedpolicy(&(resource_attr[i]), SCHED_RR);
    if (ret != 0) {
      LOG_FATAL << "Unable to set accelerator pthread scheduling policy";
      exit(1);
    }
    p1[i].sched_priority = 99;
    pthread_attr_setschedparam(&(resource_attr[i]), &p1[i]);
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

  LOG_DEBUG << "Finished accelerator initialization";
}

#endif

void initializeHardware(pthread_t *resource_handle, running_task *hardware_thread_handle,
                        pthread_mutex_t *resource_mutex) {
  LOG_DEBUG << "Performing CPU thread initialization";
  initializeCPUs(resource_handle, hardware_thread_handle, resource_mutex);
#ifdef ARM
  LOG_DEBUG << "Performing accelerator initialization";
  initializeAccelerators(resource_handle, hardware_thread_handle, resource_mutex);
#else
  LOG_DEBUG << "Skipping accelerator initialization for non-ARM build";
#endif
}

void cleanupHardware() { LOG_WARNING << "cleanupHardware not implemented"; }
