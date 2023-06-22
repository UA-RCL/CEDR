#pragma once
#include "scheduler.hpp"

void initMutexes(unsigned int resource_count, pthread_mutex_t *mutexes);

void pretendWeOnlyHaveCPUs(unsigned int resource_count, worker_thread* hardware_thread_handles);

void populateFakeCPUTask(task_nodes* task, std::string task_name, long long estimated_exec);
