#include "application.hpp"
#include "config_parse.hpp"
#include "hardware.hpp"
#include "header.hpp"
#include "runtime.hpp"
#include <execinfo.h>
#include <unistd.h>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Log.h>
#include <jarro2783/cxxopts.hpp>
#include <csignal>
#include <map>
#include <string>

/*
 * Basic handler function that uses backtrace to dump a stacktrace if a
 * particular program fault were to occur (i.e. segmentation faults)
 */
void segfault_handler(int sig) {
  void *array[10];
  size_t size;

  size = backtrace(array, 10);

  LOG_FATAL << "Error: signal " << sig;
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

int main(int argc, char **argv) {
  signal(SIGSEGV, segfault_handler);
  const unsigned int resource_count = CORE_RESOURCE_COUNT + FFT_RESOURCE_COUNT;

  std::map<std::string, void *> sharedObjectMap;
  std::map<std::string, dag_app *> applicationMap;

  cxxopts::Options options("DSSoC Emulator", "An emulation framework for running DAG-based "
                                             "applications in linux userspace");
  // clang-format off
  options.add_options()
  ("app_dir", "Directory to load applications from", cxxopts::value<std::string>())
  ("l,log-level", "Set the logging level", cxxopts::value<std::string>()->default_value("INFO"))
  ("m,mode", "Set the operational mode", cxxopts::value<std::string>()->default_value("VALIDATION"))
  ("s,scheduler", "Choose the scheduler to use", cxxopts::value<std::string>()->default_value("SIMPLE"))
  ("c,config", "File to load application configuration from", cxxopts::value<std::string>()->default_value("config.json"))
  ("h,help","Print help")
  ;
  // clang-format on
  options.positional_help("[APP_DIRECTORY]");
  options.parse_positional({"app_dir"});
  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    printf("%s\n", options.help().c_str());
    return 0;
  }
  const std::string scheduler = result["scheduler"].as<std::string>();
  const std::string config_file = result["config"].as<std::string>();

  const std::string log_str = result["log-level"].as<std::string>();
  plog::Severity log_level = plog::severityFromString(log_str.c_str());
  if (log_level == plog::Severity::none) {
    fprintf(stderr, "Unable to parse requested log level\nAcceptable options are "
                    "{\"FATAL\", \"ERROR\", \"WARN\", \"INFO\", \"DEBUG\", \"VERBOSE\"}\n");
    exit(1);
  }
  static plog::ConsoleAppender<plog::TxtFormatter> appender;
  plog::init(log_level, &appender);

  const std::string app_dir = result["app_dir"].as<std::string>();
  if (app_dir.back() != '/') {
    initializeHandlesAndApplications(sharedObjectMap, applicationMap, app_dir + "/");
  } else {
    initializeHandlesAndApplications(sharedObjectMap, applicationMap, app_dir);
  }

  printSharedObjectsFound(sharedObjectMap);
  printApplicationsLoaded(applicationMap);

  pthread_t resource_handle[resource_count];
  running_task hardware_thread_handle[resource_count];
  pthread_mutex_t resource_mutex[resource_count];

  initializeHardware(resource_handle, hardware_thread_handle, resource_mutex);

  const std::string EMULATION_MODE = result["mode"].as<std::string>();
  if (EMULATION_MODE == "PERFORMANCE") {
    const std::map<std::string, std::pair<long long, float>> config_map = parse_performance_config(config_file);
    runPerformanceMode(resource_count, applicationMap, config_map, resource_handle, hardware_thread_handle,
                       resource_mutex, scheduler);
  } else if (EMULATION_MODE == "VALIDATION") {
    const std::map<std::string, unsigned int> config_map = parse_validation_config(config_file);
    runValidationMode(resource_count, applicationMap, config_map, resource_handle, hardware_thread_handle,
                      resource_mutex, scheduler);
  } else {
    LOG_FATAL << "Unrecognized operational mode! Recognized modes are "
                 "PERFORMANCE and VALIDATION";
    exit(1);
  }

  cleanupHardware();
  closeSharedObjectHandles(sharedObjectMap);
}