#pragma once

#include "config_manager.hpp"
#include "header.hpp"
#include <pthread.h>
#include <map>
#include <string>

#define SEC2NANOSEC 1000000000
#define MS2NANOSEC (SEC2NANOSEC / 1000)
#define US2NANOSEC (MS2NANOSEC / 1000)

void launchDaemonRuntime(ConfigManager &cedr_config, pthread_t *resource_handle, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex);
