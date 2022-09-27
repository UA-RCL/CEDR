#include "config_manager.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

ConfigManager::ConfigManager() {
  // Populate the config with all of our "default" configuration options
#if defined(ARM)
  resource_array[(uint8_t) resource_type::cpu] = 3;
  resource_array[(uint8_t) resource_type::fft] = 1;
  resource_array[(uint8_t) resource_type::mmult] = 1;
  resource_array[(uint8_t) resource_type::gpu] = 0;
#else
  resource_array[(uint8_t) resource_type::cpu] = 3;
  resource_array[(uint8_t) resource_type::fft] = 0;
  resource_array[(uint8_t) resource_type::mmult] = 0;
  resource_array[(uint8_t) resource_type::gpu] = 0;
#endif

  cache_schedules = false;
  enable_queueing = true;
  use_papi = false;
  loosen_thread_permissions = false;
  fixed_periodic_injection = true;
  exit_when_idle = false;

  PAPI_Counters = std::list<std::string>();

  scheduler = "SIMPLE";
  max_parallel_jobs = 12;
  random_seed = 0;
}

ConfigManager::~ConfigManager() = default;

void ConfigManager::parseConfig(const std::string &filename) {
  std::ifstream if_stream(filename);
  if (!if_stream.is_open()) {
    LOG_ERROR << "Failed to open configuration input file " << filename;
    return;
  }
  LOG_VERBOSE << "Successfully opened configuration input file " << filename;

  nlohmann::json j;
  if_stream >> j;
  if_stream.close();

  if (j.find("Worker Threads") != j.end()) {
    LOG_VERBOSE << "Overwriting default values for number of worker threads";
    auto threads_config = j["Worker Threads"];
    for (auto &resource_type_name : resource_type_names) {
      if (threads_config.contains(resource_type_name)) {
        unsigned int resource_count = threads_config[resource_type_name];
        LOG_DEBUG << "Config > Worker Threads contains key '" << resource_type_name << "', assigning config value to " << resource_count;
        resource_array[(uint8_t) resource_type_map.at(resource_type_name)] = resource_count;
      } else {
        LOG_DEBUG << "Config > Worker Threads does not contain key '" << resource_type_name << "', skipping...";
      }
    }
  }

  if (j.find("Features") != j.end()) {
    LOG_VERBOSE << "Overwriting default values for chosen features";
    auto features_config = j["Features"];

    if (features_config.contains("Cache Schedules")) {
      LOG_DEBUG << "Config > Features contains key 'Cache Schedules', assigning config value to " << (bool)features_config["Cache Schedules"];
      cache_schedules = features_config["Cache Schedules"];
    }
    if (features_config.contains("Enable Queueing")) {
      LOG_DEBUG << "Config > Features contains key 'Enable Queueing', assigning config value to " << (bool)features_config["Enable Queueing"];
      enable_queueing = features_config["Enable Queueing"];
    }
    if (features_config.contains("Use PAPI")) {
      LOG_DEBUG << "Config > Features contains key 'Use PAPI', assigning config value to " << (bool)features_config["Use PAPI"];
      use_papi = features_config["Use PAPI"];
    }
    if (features_config.contains("Loosen Thread Permissions")) {
      LOG_DEBUG << "Config > Features contains key 'Loosen Thread Permissions', assigning config value to " << (bool)features_config["Loosen Thread Permissions"];
      loosen_thread_permissions = features_config["Loosen Thread Permissions"];
    }
    if (features_config.contains("Fixed Periodic Injection")) {
      LOG_DEBUG << "Config > Features contains key 'Fixed Periodic Injection', assigning config value to " << (bool)features_config["Fixed Periodic Injection"];
      fixed_periodic_injection = features_config["Fixed Periodic Injection"];
    }
    if (features_config.contains("Exit When Idle")) {
      LOG_DEBUG << "Config > Features contains key 'Exit When Idle', assigning config value to " << (bool)features_config["Exit When Idle"];
      exit_when_idle = features_config["Exit When Idle"];
    }
  }

  if (j.find("PAPI Counters") != j.end()) {
    LOG_VERBOSE << "Overwriting default values for papi counters";
    auto papi_counters = j["PAPI Counters"];

    for (const std::string counter : papi_counters) {
      LOG_DEBUG << "Config > PAPI Counters asks for counter '" << counter << "', adding to list of counters";
      PAPI_Counters.push_back(counter);
    }
  }

  if (j.find("Max Parallel Jobs") != j.end()) {
    LOG_DEBUG << "Config contains key 'Max Parallel Jobs', assigning config value to " << (unsigned long)j["Max Parallel Jobs"];
    max_parallel_jobs = j["Max Parallel Jobs"];
  }

  if (j.find("Scheduler") != j.end()) {
    LOG_DEBUG << "Config contains key 'Scheduler', assigning config value to " << (std::string)j["Scheduler"];
    scheduler = j["Scheduler"];
  }

  if (j.find("Random Seed") != j.end()) {
    LOG_DEBUG << "Config contains key 'Random Seed', assigning config value to " << (unsigned long)j["Random Seed"];
    random_seed = j["Random Seed"];
  }
}

unsigned int ConfigManager::getTotalResources() {
  unsigned int sum = 0;
  for (auto num_of_resource : resource_array) {
    sum += num_of_resource;
  }
  return sum;
}

unsigned int *ConfigManager::getResourceArray() { return resource_array; }

bool ConfigManager::getCacheSchedules() const { return cache_schedules; }
bool ConfigManager::getEnableQueueing() const { return enable_queueing; }
bool ConfigManager::getUsePAPI() const { return use_papi; }
bool ConfigManager::getLoosenThreadPermissions() const { return loosen_thread_permissions; }
bool ConfigManager::getFixedPeriodicInjection() const { return fixed_periodic_injection; }
bool ConfigManager::getExitWhenIdle() const { return exit_when_idle; }

std::list<std::string> &ConfigManager::getPAPICounters() { return PAPI_Counters; }

std::string &ConfigManager::getScheduler() { return scheduler; }
unsigned long ConfigManager::getMaxParallelJobs() const { return max_parallel_jobs; }
unsigned long ConfigManager::getRandomSeed() const { return random_seed; }
