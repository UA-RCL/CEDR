#include "config_manager.hpp"
#include "header.hpp"
#include "performance_monitor.hpp"
#include "runtime.hpp"
#include "threads.hpp"
#include <execinfo.h>
#include <ucontext.h> /* get REG_EIP from ucontext.h */
#include <unistd.h>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Log.h>
#include <jarro2783/cxxopts.hpp>
#include <csignal>
#include <map>
#include <string>

#if defined(USEPAPI)
#if defined(__aarch64__)
#include <papi/papi.h>
#else
#include <papi.h>
#endif
#endif

/*
 * Original segfault handler function we'd defined that simply uses backtrace to dump a stacktrace
 *
 * This version will only be called if addr2line is not present on the host system (i.e. arm dev board)
 */
void segfault_handler_fallback(int sig) {
  void *array[10];
  size_t size;

  size = backtrace(array, 10);

  backtrace_symbols_fd(array, size, STDERR_FILENO);
}

/*
 * Updated default segfault handler that attempts to provide more detailed line number information
 *
 * If addr2line is not available, it falls back to the older purely backtrace-based method
 *
 * https://stackoverflow.com/a/4611112
 */
void bt_sighandler(int sig, siginfo_t *info, void *secret) {
  void *trace[16];
  char **messages = (char **)nullptr;
  int i, trace_size = 0;
  auto *uc = (ucontext_t *)secret;

  /* Do something useful with siginfo_t */
  if (sig == SIGSEGV) {
#ifdef __aarch64__
    printf("Got signal %d, faulty address is %p, "
           "from %p\n",
           sig, info->si_addr, (void *)uc->uc_mcontext.pc);
#else
    printf("Got signal %d, faulty address is %p, "
           "from %p\n",
           sig, info->si_addr, (void *)uc->uc_mcontext.gregs[REG_RIP]);
#endif
  } else {
    printf("Got signal %d\n", sig);
  }

  // Check if addr2line is available. If not, fall back to the backtrace method
  if (system("which addr2line > /dev/null 2>&1")) {
    segfault_handler_fallback(sig);
  } else {
    trace_size = backtrace(trace, 16);

    /* overwrite sigaction with caller's address */
#ifdef __aarch64__
    trace[1] = (void *)uc->uc_mcontext.pc;
#else
    trace[1] = (void *)uc->uc_mcontext.gregs[REG_RIP];
#endif

    messages = backtrace_symbols(trace, trace_size);
    /* skip first stack frame (points here) */
    printf("[handler] Execution path:\n");

    for (i = 1; i < trace_size; ++i) {
      printf("[handler] %s\n", messages[i]);

      /* find first occurence of '(' or ' ' in message[i] and assume
       * everything before that is the file name. (Don't go beyond 0 though
       * (string terminator)*/
      size_t p = 0;
      while (messages[i][p] != '(' && messages[i][p] != ' ' && messages[i][p] != 0) {
        ++p;
      }

      char syscom[256];
      // last parameter is the filename of the symbol
      sprintf(syscom, "addr2line %p -e %.*s", trace[i], (int)p, messages[i]);

      system(syscom);
    }
  }

  exit(1);
}

int main(int argc, char **argv) {
  struct sigaction sa {};

  sa.sa_sigaction = (void (*)(int, siginfo_t *, void *)) & bt_sighandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART | SA_SIGINFO;

  sigaction(SIGSEGV, &sa, nullptr);

  cxxopts::Options options("CEDR", "An emulation framework for running DAG-based "
                                   "applications in linux userspace");
  // clang-format off
  options.add_options()
  ("l,log-level", "Set the logging level", cxxopts::value<std::string>()->default_value("INFO"))
  ("c,config-file", "Optional configuration file used to configure the CEDR runtime. Defaults are chosen for all unspecified values.", cxxopts::value<std::string>())
  ("h,help","Print help")
  ;
  // clang-format on
  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    printf("%s\n", options.help().c_str());
    return 0;
  }

  const std::string log_str = result["log-level"].as<std::string>();
  plog::Severity log_level = plog::severityFromString(log_str.c_str());
  if (log_level == plog::Severity::none && log_str != "NONE") {
    fprintf(stderr, "Unable to parse requested log level\nAcceptable options are "
                    "{\"NONE\", \"FATAL\", \"ERROR\", \"WARN\", \"INFO\", \"DEBUG\", \"VERBOSE\"}\n");
    exit(1);
  }
  static plog::ConsoleAppender<plog::TxtFormatter> appender;
  plog::init(log_level, &appender);

  ConfigManager cedr_config = ConfigManager();

  if (result.count("config-file")) {
    cedr_config.parseConfig(result["config-file"].as<std::string>());
  }

  PerfMon::initPerfLog();

  const unsigned int totalResources = cedr_config.getTotalResources();

  pthread_t resource_handle[totalResources];
  worker_thread hardware_thread_handle[totalResources];
  pthread_mutex_t resource_mutex[totalResources];

#if defined(USEPAPI)
  if (cedr_config.getUsePAPI()) {
    int papiRet = PAPI_library_init(PAPI_VER_CURRENT);
    if (papiRet != PAPI_VER_CURRENT) {
      LOG_WARNING << "Unable to initialize PAPI, there is a library version mismatch (expected " << PAPI_VER_CURRENT << ", got " << papiRet << ")";
      if (papiRet < 0) {
        LOG_WARNING << "PAPI error: " << std::string(PAPI_strerror(papiRet));
      }
    } else {
      LOG_DEBUG << "PAPI initialized, dumping CSV header to the performance log";
      std::stringstream sstream;
      size_t idx = 0;
      sstream << "Resource Name"
              << ", "
              << "Task Name"
              << ", ";
      for (const auto &papiCounter : cedr_config.getPAPICounters()) {
        sstream << papiCounter;
        if (idx < cedr_config.getPAPICounters().size() - 1) {
          sstream << ", ";
        }
        idx++;
      }
      LOG_INFO_(PerfMon::LoggerId) << sstream.str();
    }
    papiRet = PAPI_thread_init(pthread_self);
    if (papiRet != PAPI_OK) {
      LOG_WARNING << "Unable to initialize PAPI threading support, worker threads might conflict and cause CEDR to crash.";
    } else {
      LOG_DEBUG << "PAPI threading initialized";
    }
  }
#endif

  initializeThreads(cedr_config, resource_handle, hardware_thread_handle, resource_mutex);

  launchDaemonRuntime(cedr_config, resource_handle, hardware_thread_handle, resource_mutex);

  cleanupThreads(cedr_config);
}
