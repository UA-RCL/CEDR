#pragma once

#include "header.hpp"
#include <pthread.h>
#include <map>
#include <string>

#define SEC2NANOSEC 1000000000
#define MS2NANOSEC (SEC2NANOSEC / 1000)

// Total experiment runtime in milliseconds
const unsigned int MAX_RUNTIME = 1000;

void runValidationMode(unsigned int resource_count, std::map<std::string, dag_app *> &applicationMap,
                       const std::map<std::string, unsigned int> &config_map, pthread_t *resource_handle,
                       running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex, std::string scheduler,
                       bool cache_schedules);

void runPerformanceMode(unsigned int resource_count, std::map<std::string, dag_app *> &applicationMap,
                        const std::map<std::string, std::pair<long long, float>> &config_map,
                        pthread_t *resource_handle, running_task *hardware_thread_handle,
                        pthread_mutex_t *resource_mutex, std::string scheduler, bool cache_schedules);
#ifdef DAEMON
void runPerformanceMode_in_daemon(unsigned int resource_count, pthread_t *resource_handle,
                                  running_task *hardware_thread_handle, pthread_mutex_t *resource_mutex,
                                  std::string scheduler, bool cache_schedules);
#endif
