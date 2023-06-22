#pragma once

#include "config_manager.hpp"
#include "header.hpp"
#include <pthread.h>

#define MAX_ARGS 79

void *hardware_thread(void *ptr);

void initializeThreads(ConfigManager &cedr_config, pthread_t *resource_handle, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex);

void cleanupThreads(ConfigManager &cedr_config);
