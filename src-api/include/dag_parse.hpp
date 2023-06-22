#pragma once

#include "header.hpp"
#include <map>

cedr_app *parse_binary(const std::string &filename, std::map<std::string, void *> &sharedObjectMap);

