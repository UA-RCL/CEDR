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
  const unsigned int resource_count = TOTAL_RESOURCE_COUNT;

  std::map<std::string, void *> sharedObjectMap;
  std::map<std::string, dag_app *> applicationMap;

  cxxopts::Options options("CEDR", "An emulation framework for running DAG-based "
                                   "applications in linux userspace");
  // clang-format off
  options.add_options()
  ("l,log-level", "Set the logging level", cxxopts::value<std::string>()->default_value("INFO"))
  ("s,scheduler", "Choose the scheduler to use", cxxopts::value<std::string>()->default_value("SIMPLE"))
  ("cache-schedules", "Once a node is scheduled, future iterations of that node will use the same decision", cxxopts::value<bool>()->default_value("false"))
  ("h,help","Print help")
  ;
  // clang-format on
  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    printf("%s\n", options.help().c_str());
    return 0;
  }
  const std::string scheduler = result["scheduler"].as<std::string>();
  const bool cache_schedules = result["cache-schedules"].as<bool>();

  const std::string log_str = result["log-level"].as<std::string>();
  plog::Severity log_level = plog::severityFromString(log_str.c_str());
  if (log_level == plog::Severity::none) {
    fprintf(stderr, "Unable to parse requested log level\nAcceptable options are "
                    "{\"FATAL\", \"ERROR\", \"WARN\", \"INFO\", \"DEBUG\", \"VERBOSE\"}\n");
    exit(1);
  }
  static plog::ConsoleAppender<plog::TxtFormatter> appender;
  plog::init(log_level, &appender);

  pthread_t resource_handle[resource_count];
  running_task hardware_thread_handle[resource_count];
  pthread_mutex_t resource_mutex[resource_count];

  initializeHardware(resource_handle, hardware_thread_handle, resource_mutex);

  runPerformanceMode_in_daemon(resource_count, resource_handle, hardware_thread_handle, resource_mutex, scheduler,
                               cache_schedules);

  cleanupHardware();
  closeSharedObjectHandles(sharedObjectMap);
}
