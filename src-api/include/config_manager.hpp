#pragma once

#include "header.hpp"
#include <list>
#include <string>

class ConfigManager {
  // When unit testing, we need the flexibility to tweak the various config fields individually
#if defined(CEDR_UNIT_TESTING)
public:
#else
private:
#endif
  unsigned int resource_array[(uint8_t) resource_type::NUM_RESOURCE_TYPES]{};

  bool cache_schedules;
  bool enable_queueing;
  bool use_papi;
  bool loosen_thread_permissions;
  bool fixed_periodic_injection;
  bool exit_when_idle;

  long long dash_exec_times[api_types::NUM_API_TYPES][(uint8_t) resource_type::NUM_RESOURCE_TYPES];
  std::list<std::string> PAPI_Counters;

  std::string scheduler;
  // TODO: the logic for respecting this is not implemented
  unsigned long max_parallel_jobs;
  unsigned long random_seed;

  std::list<std::string> dash_binary_paths;
public:
  ConfigManager();
  ~ConfigManager();

  void parseConfig(const std::string &filename);

  unsigned int getTotalResources();

  unsigned int *getResourceArray();

  bool getCacheSchedules() const;
  bool getEnableQueueing() const;
  bool getUsePAPI() const;
  bool getLoosenThreadPermissions() const;
  bool getFixedPeriodicInjection() const;
  bool getExitWhenIdle() const;

  long long getDashExecTime(const api_types api, const resource_type resource);

  std::list<std::string> &getPAPICounters();

  std::string &getScheduler();
  unsigned long getMaxParallelJobs() const;
  unsigned long getRandomSeed() const;

  std::list<std::string> &getDashBinaryPaths();
};
