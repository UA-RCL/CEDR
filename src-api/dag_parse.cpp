#include <dirent.h>
#include <dlfcn.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <list>
#include <linux/limits.h>

#include <plog/Log.h>
#include <nlohmann/json.hpp>

#include "dag_parse.hpp"

#ifdef USELIEF
#include "LIEF/LIEF.hpp"
#endif

using json = nlohmann::json;

void *openHandle(std::map<std::string, void *> &sharedObjectMap, const std::string &sharedObjPath, const std::string &sharedObjName) {
  void *dl_handle = dlopen(sharedObjPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!dl_handle) {
    char currentWorkingDir[PATH_MAX];
    getcwd(currentWorkingDir, PATH_MAX);
    LOG_FATAL << "Failed to open shared object file!" << std::endl << std::string(dlerror());
    LOG_FATAL << "My current working directory is " << std::string(currentWorkingDir);
    return nullptr; //exit(1);
  }
  sharedObjectMap[sharedObjName] = dl_handle;
  return dl_handle;
}

cedr_app *parse_binary(const std::string &filename, std::map<std::string, void *> &sharedObjectMap){
  // INFO: Open the shared object and grab a handle to main function
  //  Return an app structure with proper *dlhandle  assigned
  void *dl_handle;

  const std::string shared_object_name = filename.substr(filename.find_last_of("/")+1);
  const std::string app_name = shared_object_name.substr(0, shared_object_name.find_last_of("."));
  LOG_VERBOSE << "[DAG_PARSE] Received app name is " << app_name << " with shared_object " << shared_object_name << " at path " << filename;
  auto handleIsPresent = sharedObjectMap.find(shared_object_name);
  if (handleIsPresent == sharedObjectMap.end()) {
    LOG_INFO << "I have not opened this shared object before. Looking for it at \"" << filename << "\"";
    dl_handle = openHandle(sharedObjectMap, filename, shared_object_name);
    if (dl_handle == nullptr) {
      LOG_ERROR << "Unable to find shared object " << shared_object_name << " associated with application " << app_name;
      return nullptr;
    }
  } else {
    dl_handle = handleIsPresent->second;
  }

  void (*lib_main)(void*);
  lib_main = (void(*)(void*))dlsym(dl_handle, "main");

  if (lib_main == NULL) {
    LOG_ERROR << "Unable to get function handle for main function in " << shared_object_name << "file!";
    return nullptr;
    //return -1;
  }

  cedr_app *myapp;
  myapp = (cedr_app *)calloc(1, sizeof(cedr_app));
  //myapp->app_id;    // INFO: Not modified here, probably in runtime
  strncpy(myapp->app_name, app_name.c_str(), sizeof(myapp->app_name));  // TODO: Convert everything to std::string please
  myapp->app_name[sizeof(myapp->app_name) - 1] = '\0';  // INFO: Ensure that the string terminates in null byte
  myapp->dlhandle = dl_handle;
  myapp->main_func_handle = lib_main;

  return myapp;
}
