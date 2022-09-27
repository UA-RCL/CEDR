#pragma once

#include "config_manager.hpp"
#include "header.hpp"
#include <pthread.h>
#include <deque>
#include <string>

void performScheduling(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                       uint32_t &free_resource_count);
