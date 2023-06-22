#pragma once

#include "header.hpp"
#include <map>

dag_app *parse_dag_and_binary(const std::string &filename, std::map<std::string, void *> &sharedObjectMap, bool readBinaryDAG = false);
