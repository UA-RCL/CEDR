#include "performance_monitor.hpp"

namespace PerfMon {
void initPerfLog(const std::string &filename) {
  static plog::RollingFileAppender<plog::CsvFormatter> csvAppender(filename.c_str());
  plog::init<PerfMon::LoggerId>(plog::info, &csvAppender);
}
} // namespace PerfMon