#include "application.hpp"
#include "dag_parse.hpp"
#include <dirent.h>
#include <dlfcn.h>
#include <plog/Log.h>
#include <map>

void initializeHandles(std::map<std::string, void *> &sharedObjectMap, const std::string &app_dir) {
  DIR *dirp = opendir(app_dir.c_str());
  if (dirp == nullptr) {
    LOG_FATAL << "Failed to open application directory: " << app_dir;
    exit(1);
  }
  struct dirent *dp;
  const std::string target_suffix(".so");
  while ((dp = readdir(dirp)) != nullptr) {
    std::string curr_file(dp->d_name);
    // Check if the given string ends with the target filename suffix
    if (curr_file.size() < target_suffix.size()) {
      continue;
    }
    if (curr_file.substr(curr_file.size() - target_suffix.size()) == target_suffix) {
      std::string full_path = app_dir + curr_file;
      void *dl_handle = dlopen(full_path.c_str(), RTLD_LAZY);
      if (!dl_handle) {
        LOG_FATAL << "Failed to open shared object file!" << std::endl << std::string(dlerror());
        exit(1);
      }
      sharedObjectMap[curr_file] = dl_handle;
    }
  }
  closedir(dirp);
}

void initializeApplications(std::map<std::string, void *> &sharedObjectMap,
                            std::map<std::string, dag_app *> &applicationMap, const std::string &app_dir) {
  DIR *dirp = opendir(app_dir.c_str());
  if (dirp == nullptr) {
    LOG_FATAL << "Failed to open application directory: " << app_dir;
    exit(1);
  }
  struct dirent *dp;
#ifdef USELIEF
  const std::string target_suffix(".inst");
#else
  const std::string target_suffix(".json");
#endif
  while ((dp = readdir(dirp)) != NULL) {
    std::string curr_file(dp->d_name);
    // Check if the given string ends with the target filename suffix
    if (curr_file.size() < target_suffix.size()) {
      continue;
    }
    if (curr_file.substr(curr_file.size() - target_suffix.size()) == target_suffix) {
      std::string full_path = app_dir + curr_file;
#ifdef USELIEF
      dag_app *this_app = parse_dag_file(full_path.c_str(), sharedObjectMap, true);
#else
      dag_app *this_app = parse_dag_file(full_path, sharedObjectMap, false, nullptr);
#endif
      if (this_app == nullptr) {
        LOG_ERROR << "Unable to initialize application associated with " << curr_file;
      } else {
        applicationMap[std::string(this_app->app_name)] = this_app;
      }
    }
  }
  closedir(dirp);
}

/*
    Initializes the map of shared object name to dlopen handles
    This ensures that all function pointers retrieved with dlopen/dlsym remain
   valid for the lifetime of the program
*/
void initializeHandlesAndApplications(std::map<std::string, void *> &sharedObjectMap,
                                      std::map<std::string, dag_app *> &applicationMap, const std::string &app_dir) {
  LOG_DEBUG << "Initializing shared object and application handles";
  initializeHandles(sharedObjectMap, app_dir);
  initializeApplications(sharedObjectMap, applicationMap, app_dir);
  LOG_DEBUG << "Finished initializing shared object and application handles";
}

void closeSharedObjectHandles(std::map<std::string, void *> &sharedObjectMap) {
  for (const auto &entry : sharedObjectMap) {
    dlclose(entry.second);
  }
}

void printSharedObjectsFound(std::map<std::string, void *> &sharedObjectMap) {
  for (const auto &kv : sharedObjectMap) {
    LOG_INFO << "We have a handle for shared object with (k, v): (" << kv.first << ", " << kv.second << ")";
  }
}

void printApplicationsLoaded(std::map<std::string, dag_app *> &applicationMap) {
  for (const auto &kv : applicationMap) {
    LOG_INFO << "We have an application pointer with (k, v): (" << kv.first << ", " << kv.second << ")";

    IF_LOG(plog::debug) {
      LOG_DEBUG << "Dumping attributes of this application...";
      printf("\x1B[96m"); // set color to cyan to match with logger
      dag_app *myapp = kv.second;
      task_nodes *node_list = myapp->head_node;

      printf("Parsed application %s\napp_id: %d\narrival_time: %lu\nnum_tasks: "
             "%d\nContaining the following tasks:\n\n",
             myapp->app_name, myapp->app_id, myapp->arrival_time, myapp->task_count);

      for (int i = 0; i < myapp->task_count; i++) {
        printf("Task id: %d (name: %s) with %d predecessors, and %d "
               "successors, and %d supported resources\n",
               node_list[i].task_id, node_list[i].task_name, node_list[i].pred_count, node_list[i].succ_count,
               node_list[i].supported_resource_count);

        printf("Arguments: [");
        for (int j = 0; j < node_list[i].args.size(); j++) {
          printf("%s", node_list[i].args.at(j)->name.c_str());
          if (j < node_list[i].args.size() - 1) {
            printf(", ");
          }
        }
        printf("];\n");

        printf("Predecessors: [");
        for (int j = 0; j < node_list[i].pred_count; j++) {
          printf("%d", node_list[i].pred[j]->task_id);
          if (j < node_list[i].pred_count - 1) {
            printf(", ");
          }
        }
        printf("];\n");

        printf("Successors: [");
        for (int j = 0; j < node_list[i].succ_count; j++) {
          printf("%d", node_list[i].succ[j]->task_id);
          if (j < node_list[i].succ_count - 1) {
            printf(", ");
          }
        }
        printf("];\n");

        printf("Supported resources: [");
        for (int j = 0; j < node_list[i].supported_resource_count; j++) {
          printf("(%s, %f)", node_list[i].supported_resources[j], node_list[i].estimated_execution[j]);
          if (j < node_list[i].supported_resource_count - 1) {
            printf(", ");
          }
        }
        printf("];\n\n");
      }
      printf("\x1B[0m\x1B[0K"); // reset coloring
    }
  }
}