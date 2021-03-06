#pragma once

#include "header.hpp"
#include <map>

dag_app *parse_dag_file(const std::string &filename, std::map<std::string, void *> &sharedObjectMap, bool readBinaryDAG,
                        dag_app *app_param = nullptr);
