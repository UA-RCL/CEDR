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

  std::list<std::string> PAPI_Counters;

  std::string scheduler;
  // TODO: the logic for respecting this is not implemented
  unsigned long max_parallel_jobs;
  unsigned long random_seed;

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

  std::list<std::string> &getPAPICounters();

  std::string &getScheduler();
  unsigned long getMaxParallelJobs() const;
  unsigned long getRandomSeed() const;
};
