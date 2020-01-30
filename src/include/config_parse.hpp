#pragma once

#include <map>

std::map<std::string, unsigned int> parse_validation_config(const std::string &filename);
std::map<std::string, std::pair<long long, float>> parse_performance_config(const std::string &filename);
std::map<std::string, unsigned int> parse_trace_config(const std::string &filename);