#pragma once

#include "header.hpp"
#include <pthread.h>

#define MAX_ARGS 20

void *hardware_thread(void *ptr);

void initializeHardware(pthread_t *resource_handle, running_task *hardware_thread_handle,
                        pthread_mutex_t *resource_mutex);

void cleanupHardware();