#pragma once

#include <plog/Log.h>
#include <cstdint>
#include <map>
#include <string>

namespace PerfMon {
enum {
  // 0 is not specified here and is used for the default logger instance
  LoggerId = 1
};

void initPerfLog(const std::string &filename = "perf_stats.csv");
} // namespace PerfMon