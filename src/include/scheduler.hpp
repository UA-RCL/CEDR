#pragma once

#include "header.hpp"
#include <pthread.h>
#include <deque>
#include <string>

#define SCHED_POLICY "MET"
#define RAND_SEED 0

void performScheduling(std::deque<task_nodes *> &ready_queue, unsigned int resource_count,
                       running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                       std::string sched_policy = std::string(SCHED_POLICY), bool cache_schedules = false);
