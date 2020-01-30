#include "config_parse.hpp"
#include <plog/Log.h>
#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

std::map<std::string, unsigned int> parse_validation_config(const std::string &filename) {
  std::ifstream if_stream(filename);
  std::map<std::string, unsigned int> config_map;
  if (!if_stream.is_open()) {
    LOG_ERROR << "Failed to open configuration input file " << filename;
    return config_map;
  }

  json j;
  if_stream >> j;
  if_stream.close();

  if (j.find("validation_config") == j.end()) {
    LOG_ERROR << "Validation mode was requested, but config file " << filename
              << " does not have a validation_config key to parse";
    return config_map;
  }

  for (json::iterator it1 = j["validation_config"].begin(); it1 != j["validation_config"].end(); ++it1) {
    const std::string &application = it1.key();
    unsigned int run_instances = *it1;
    config_map[application] = run_instances;
  }

  return config_map;
}

std::map<std::string, std::pair<long long, float>> parse_performance_config(const std::string &filename) {
  std::ifstream if_stream(filename);
  std::map<std::string, std::pair<long long, float>> config_map;
  if (!if_stream.is_open()) {
    LOG_ERROR << "Failed to open configuration input file " << filename;
    return config_map;
  }

  json j;
  if_stream >> j;
  if_stream.close();

  if (j.find("performance_config") == j.end()) {
    LOG_ERROR << "Performance mode was requested, but config file " << filename
              << " does not have a performance_config key to parse";
    return config_map;
  }

  for (json::iterator it1 = j["performance_config"].begin(); it1 != j["performance_config"].end(); ++it1) {
    const std::string &application = it1.key();
    long long period = std::numeric_limits<long long>::max();
    float prob = 0.0f;

    if (it1->find("period") == it1->end()) {
      LOG_ERROR << "While parsing performance mode config, application " << application
                << " does not supply an application injection period. Assigning to max possible value";
    } else {
      period = (*it1)["period"];
    }

    if (it1->find("probability") == it1->end()) {
      LOG_ERROR << "While parsing performance mode config, application " << application
                << " does not supply an injection probability. Assigning to 0";
    } else {
      prob = (*it1)["probability"];
    }
    config_map[application] = std::pair<long long, float>(period, prob);
  }

  return config_map;
}

std::map<std::string, unsigned int> parse_trace_config(const std::string &filename) {
  std::ifstream if_stream(filename);
  std::map<std::string, unsigned int> config_map;
  if (!if_stream.is_open()) {
    LOG_ERROR << "Failed to open configuration input file " << filename;
    return config_map;
  }

  json j;
  if_stream >> j;
  if_stream.close();

  if (j.find("trace_config") == j.end()) {
    LOG_ERROR << "Trace execution mode was requested, but config file " << filename
              << " does not have a trace_config key to parse";
    return config_map;
  }

  for (json::iterator it1 = j["trace_config"].begin(); it1 != j["trace_config"].end(); ++it1) {
    const std::string &application = it1.key();
    unsigned int injection_time = *it1;
    config_map[application] = injection_time;
  }

  return config_map;
}
