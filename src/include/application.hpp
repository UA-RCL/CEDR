#pragma once

#include "header.hpp"
#include <map>
#include <string>

void initializeHandlesAndApplications(std::map<std::string, void *> &sharedObjectMap,
                                      std::map<std::string, dag_app *> &applicationMap, const std::string &app_dir);

void closeSharedObjectHandles(std::map<std::string, void *> &sharedObjectMap);

void printSharedObjectsFound(std::map<std::string, void *> &sharedObjectMap);

void printApplicationsLoaded(std::map<std::string, dag_app *> &applicationMap);