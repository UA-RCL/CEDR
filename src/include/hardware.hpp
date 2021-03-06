#pragma once

#include "header.hpp"
#include <pthread.h>

#define MAX_ARGS 59

void *hardware_thread(void *ptr);

void initializeHardware(pthread_t *resource_handle, running_task *hardware_thread_handle,
                        pthread_mutex_t *resource_mutex, bool loosen_thread_permissions = false);

void cleanupHardware();
