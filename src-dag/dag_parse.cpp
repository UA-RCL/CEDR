#include <dirent.h>
#include <dlfcn.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <list>

#include <plog/Log.h>
#include <nlohmann/json.hpp>

#include "dag_parse.hpp"

#ifdef USELIEF
#include "LIEF/LIEF.hpp"
#endif

using json = nlohmann::json;

void *openHandle(std::map<std::string, void *> &sharedObjectMap, const std::string &sharedObjPath, const std::string &sharedObjName) {
  void *dl_handle = dlopen(sharedObjPath.c_str(), RTLD_LAZY);
  if (!dl_handle) {
    char currentWorkingDir[PATH_MAX];
    getcwd(currentWorkingDir, PATH_MAX);
    LOG_FATAL << "Failed to open shared object file!" << std::endl << std::string(dlerror());
    LOG_FATAL << "My current working directory is " << std::string(currentWorkingDir);
    exit(1);
  }
  sharedObjectMap[sharedObjName] = dl_handle;
  return dl_handle;
}

dag_app *parse_dag_and_binary(const std::string &filename, std::map<std::string, void *> &sharedObjectMap, bool readBinaryDAG /* = false */) {

  json j;
  void *dl_handle;
  void *func_handle;
  char *dl_err;

  std::list<int> headnode_indices;
  std::list<int> tailnode_indices;

  const std::string basepath = filename.substr(0, filename.find_last_of('/'));

  if (readBinaryDAG) {
#ifdef USELIEF
#ifdef ARM
    LOG_ERROR << "LIEF functionality not ported to ARM";
    return nullptr;
#else
    // TODO: This likely has regressed in functionality as it expects the filename to itself be an ELF binary, which we
    // never are currently passing in
    std::unique_ptr<LIEF::ELF::Binary> elf = LIEF::ELF::Parser::parse(std::string(filename));
    std::vector<uint8_t> descr = elf->get(LIEF::ELF::NOTE_TYPES::NT_UNKNOWN).description();
    std::string dag_str(descr.begin(), descr.end());
    std::stringstream sstream(dag_str);
    sstream >> j;
#endif
#endif
  } else {
    std::ifstream if_stream(filename);
    if (!if_stream.is_open()) {
      LOG_ERROR << "Failed to open input file " << filename;
      return nullptr;
    }

    if_stream >> j;
    if_stream.close();
  }

  const std::string app_name = j["AppName"];
  const std::string shared_object_name = j["SharedObject"];
  json variables = j["Variables"];

  stream_mode_t check_stream_app = stream_mode_t::non_streaming;
  int max_stream_frame_count = 0;
  if (j.find("Stream_enable") != j.end()) {
    if (stream_mode_map.find(j["Stream_enable"]) != stream_mode_map.end()) {
      check_stream_app = stream_mode_map.at(j["Stream_enable"]);
      LOG_INFO << "Selected streaming mode '" << stream_mode_names[check_stream_app] << "' for application '" << app_name << "'";
    } else {
      LOG_ERROR << "Invalid streaming mode '" << std::string(j["Stream_enable"]) << "' used in DAG(.json). Options: 'none', 'single buffer', 'double buffer', 'sequential'";
      LOG_WARNING << "Using streaming mode 'none' instead!";
    }
  }
  if (j.find("Max_stream_frame_count") != j.end()) {
    max_stream_frame_count = j["Max_stream_frame_count"];
  }

  auto handleIsPresent = sharedObjectMap.find(shared_object_name);
  if (handleIsPresent == sharedObjectMap.end()) {
    std::string fullpath(basepath);
    fullpath.append("/").append(shared_object_name);
    LOG_INFO << "I have not opened this shared object before. Looking for it at \"" << fullpath << "\"";
    dl_handle = openHandle(sharedObjectMap, fullpath, shared_object_name);
    // LOG_ERROR << "Unable to find shared object " << shared_object_name << " associated with application " <<
    // app_name; return nullptr;
  } else {
    dl_handle = handleIsPresent->second;
  }

  j = j["DAG"];
  const int num_nodes = j.size();

  dag_app *myapp;
  myapp = (dag_app *)calloc(1, sizeof(dag_app));
  task_nodes *node_list = (task_nodes *)calloc(num_nodes, sizeof(task_nodes));

  auto variable_map = std::map<std::string, variable *>();

  myapp->variables = variable_map;
  for (json::iterator it1 = variables.begin(); it1 != variables.end(); ++it1) {
    std::string name = std::string(it1.key());
    auto num_bytes = (unsigned int)(*it1)["bytes"];
    LOG_DEBUG << "Allocating variable " << name << " with " << num_bytes << " bytes";
    auto *var = (variable *)calloc(1, sizeof(variable));
    var->name = name;
    var->num_bytes = num_bytes;
    var->heap_ptr = calloc(num_bytes, sizeof(uint8_t));

    if (it1->find("val") != it1->end()) {
      std::vector<uint8_t> bytes = (*it1)["val"];
      if (!bytes.empty()) {
        LOG_DEBUG << "This variable has set of bytes to initialize the value with";
        LOG_DEBUG << "They are: ";
        std::string str;
        for (auto byte : bytes) {
          str += std::to_string(byte) + " ";
        }
        LOG_DEBUG << str;
        if (bytes.size() != num_bytes) {
          LOG_WARNING << "While allocating variable " << name << " with " << num_bytes << " bytes, " << bytes.size()
                      << " bytes were given to initialize with. Ignoring provided value and leaving var 0-initialized.";
        } else {
          LOG_DEBUG << "Copying them over now";
          memcpy(var->heap_ptr, bytes.data(), num_bytes);
        }
      }
    }

    if (it1->find("is_ptr") != it1->end()) {
      bool isPtr = (*it1)["is_ptr"];
      if (isPtr) {
        var->is_ptr_var = true;
        if (it1->find("ptr_alloc_bytes") != it1->end()) {
          uint64_t ptrBytes = (*it1)["ptr_alloc_bytes"];
          LOG_DEBUG << "This variable is a pointer type and it is requesting allocation of " << ptrBytes << " bytes";
          if (sizeof(uint64_t) != num_bytes) {
            LOG_WARNING << "While allocating memory for pointer variable " << name << ", the variable only requested " << num_bytes << " bytes of storage, but the pointer is "
                        << sizeof(uint64_t) << " bytes in size."
                        << " Leaving the pointer as a null pointer";
          } else {
            auto *ptr_loc = (uint64_t *)calloc(ptrBytes, sizeof(uint8_t));
            memcpy(var->heap_ptr, &ptr_loc, num_bytes);
          }
        } else {
          LOG_WARNING << "Variable " << name << " is of pointer type, but it does not request any bytes to be allocated. "
                      << "Leaving as a null pointer";
        }
      }
    }

    myapp->variables[var->name] = var;
  }

  myapp->head_node = node_list;
  myapp->task_count = num_nodes;
  myapp->stream_enable = check_stream_app;
  myapp->max_stream_frame_count = max_stream_frame_count;
  strncpy(myapp->app_name, app_name.c_str(), sizeof(myapp->app_name));
  if (strlen(app_name.c_str()) >= sizeof(myapp->app_name)) {
    myapp->app_name[sizeof(myapp->app_name) - 1] = '\0';
  }
  // TODO: For applications with multiple shared objects, holding on to just a single handle might not be sufficient?
  myapp->dlhandle = dl_handle;

  // Build up the list of each node
  unsigned int task_id = 0;
  bool usingCustomTaskIds = false;
  std::map<std::string, unsigned int> node_to_idx;
  for (json::iterator it1 = j.begin(); it1 != j.end(); ++it1) {
    json curr_elem = j[it1.key()];
    // Each node has arguments, predecessors, successors, and platforms lists
    json arguments = curr_elem["arguments"];
    json pred_list = curr_elem["predecessors"];
    json succ_list = curr_elem["successors"];
    json plat_list = curr_elem["platforms"];

    if (it1 == j.begin()) {
      if (curr_elem.find("task_id") != curr_elem.end()) {
        LOG_DEBUG << "While parsing " << app_name << ", I found a custom task id in the first element. "
                  << "Assuming all elements use custom task ids";
        usingCustomTaskIds = true;
      }
    } else {
      if (curr_elem.find("task_id") != curr_elem.end() && !usingCustomTaskIds) {
        LOG_ERROR << "Not currently parsing with custom task ids, but found an element (" << it1.key()
                  << ") that has a custom task id. Ignoring, and this application will be parsed incorrectly!";
      } else if (curr_elem.find("task_id") == curr_elem.end() && usingCustomTaskIds) {
        LOG_ERROR << "Currently parsing with custom task ids, but found an element (" << it1.key()
                  << ") that does not have a custom task id. This application will be parsed incorrectly!";
      }
    }

    if (usingCustomTaskIds) {
      task_id = curr_elem["task_id"];
    }

    node_list[task_id].args = std::vector<variable *>();
    for (json::iterator arg_itr = arguments.begin(); arg_itr != arguments.end(); ++arg_itr) {
      std::string var_name = std::string(*arg_itr);
      if (myapp->variables.find(var_name) == myapp->variables.end()) {
        LOG_ERROR << "While parsing node " << task_id << ", I encountered an unknown argument variable " << var_name;
      } else {
        node_list[task_id].args.push_back(myapp->variables[var_name]);
      }
    }

    if (node_to_idx.find(it1.key()) != node_to_idx.end()) {
      LOG_ERROR << "Node with duplicate key encountered in DAG parsing: key \"" << it1.key() << "\" has already been encountered";
      LOG_ERROR << "Continuing parsing, but the resulting application may be "
                   "incorrect...";
    } else {
      node_to_idx[it1.key()] = task_id;
    }

    node_list[task_id].task_id = task_id;
    node_list[task_id].app_pnt = myapp;
    const std::string task_name_str = app_name + "_" + it1.key();
    node_list[task_id].task_name = task_name_str;

    // Finding head node
    if (pred_list.size() == 0){
      LOG_DEBUG << "Found head node " << std::string(node_list[task_id].task_name)
                << " at index " << task_id;
      node_list[task_id].head_node = true;
      headnode_indices.push_back(task_id);
    }
    node_list[task_id].pred_count = pred_list.size();
    node_list[task_id].pred = (task_nodes **)calloc(pred_list.size(), sizeof(task_nodes *));

    // Finding tail node
    if (succ_list.size() == 0){
      LOG_DEBUG << "Found tail node " << std::string(node_list[task_id].task_name)
                << " at index " << task_id;
      node_list[task_id].tail_node = true;
      tailnode_indices.push_back(task_id);
    }
    node_list[task_id].succ_count = succ_list.size();
    node_list[task_id].succ = (task_nodes **)calloc(succ_list.size(), sizeof(task_nodes *));

    node_list[task_id].supported_resources = std::set<resource_type>();

    node_list[task_id].actual_resource_cluster_idx = -1;
    node_list[task_id].assigned_resource_name = "unassigned";

    if (!usingCustomTaskIds) {
      task_id++;
    }
  }

  // Iterate over again and link the nodes together
  int idx;
  task_id = 0;
  for (json::iterator it1 = j.begin(); it1 != j.end(); ++it1) {
    json curr_elem = j[it1.key()];

    json pred_list = j[it1.key()]["predecessors"];
    json succ_list = j[it1.key()]["successors"];
    json plat_list = j[it1.key()]["platforms"];

    if (usingCustomTaskIds) {
      task_id = curr_elem["task_id"];
    }

    idx = 0;
    // for sequentia streaming (3) mode, assign tail nodes of app. as
    // predecessors of head nodes of app.
    if ((myapp->stream_enable == 3) && node_list[task_id].head_node){
      node_list[task_id].pred_count = tailnode_indices.size();
      free(node_list[task_id].pred);
      node_list[task_id].pred = (task_nodes **)calloc(tailnode_indices.size(), sizeof(task_nodes *));
      for (auto tail_idx = tailnode_indices.begin(); tail_idx != tailnode_indices.end(); ++tail_idx, ++idx){
        node_list[task_id].pred[idx] = &node_list[*tail_idx];
      }
    } else {
      for (json::iterator pred_itr = pred_list.begin(); pred_itr != pred_list.end(); ++pred_itr, ++idx) {
        unsigned int pred_idx = node_to_idx[std::string((*pred_itr)["name"])];
        node_list[task_id].pred[idx] = &node_list[pred_idx];
      }
    }

    idx = 0;
    // for sequential streaming (3) mode, assign head nodes of app. as
    // successors of tail nodes of app.
    if ((node_list[task_id].app_pnt->stream_enable == 3) && node_list[task_id].tail_node){
      node_list[task_id].succ_count = headnode_indices.size();
      free(node_list[task_id].succ);
      node_list[task_id].succ = (task_nodes **)calloc(headnode_indices.size(), sizeof(task_nodes *));
      for (auto head_idx = headnode_indices.begin(); head_idx != headnode_indices.end(); ++head_idx, ++idx){
        node_list[task_id].succ[idx] = &node_list[*head_idx];
      }
    }
    else {
      for (json::iterator succ_itr = succ_list.begin(); succ_itr != succ_list.end(); ++succ_itr, ++idx) {
        unsigned int succ_idx = node_to_idx[std::string((*succ_itr)["name"])];
        node_list[task_id].succ[idx] = &node_list[succ_idx];
      }
    }

    idx = 0;
    for (json::iterator plat_itr = plat_list.begin(); plat_itr != plat_list.end(); ++plat_itr, ++idx) {
      std::string platform = std::string((*plat_itr)["name"]);
      std::string runfunc = std::string((*plat_itr)["runfunc"]);
      const resource_type resourceType = resource_type_map.at(platform);

      node_list[task_id].supported_resources.insert(resourceType);
      node_list[task_id].estimated_execution[(uint8_t) resourceType] = (long long)((float)(*plat_itr)["nodecost"] * USEC2NANOSEC);

      if (plat_itr->find("shared_object") != plat_itr->end()) {
        LOG_INFO << "Node " << it1.key() << " has platform " << platform << " that uses a custom shared object";
        std::string custom_shared_object_name = std::string((*plat_itr)["shared_object"]);
        auto customHandleIsPresent = sharedObjectMap.find(custom_shared_object_name);
        void *custom_dl_handle;
        if (customHandleIsPresent == sharedObjectMap.end()) {
          std::string fullpath(basepath);
          fullpath.append("/").append(custom_shared_object_name);
          LOG_INFO << "Looking for new shared object at \"" << fullpath << "\"";
          custom_dl_handle = openHandle(sharedObjectMap, fullpath, custom_shared_object_name);
          func_handle = dlsym(custom_dl_handle, runfunc.c_str());
        } else {
          custom_dl_handle = customHandleIsPresent->second;
          func_handle = dlsym(custom_dl_handle, runfunc.c_str());
        }
      } else {
        func_handle = dlsym(dl_handle, runfunc.c_str());
      }

      if ((dl_err = dlerror()) != nullptr) {
        LOG_ERROR << "Error while assigning runfunc handle: " << std::endl << std::string(dl_err);
      } else {
        node_list[task_id].run_funcs[(uint8_t) resourceType] = func_handle;
      }
    }
    if (!usingCustomTaskIds) {
      task_id++;
    }
  }
  LOG_VERBOSE << "For app. " << std::string(myapp->app_name) << ", number of (head, tail) nodes found ("
            <<headnode_indices.size() << ", " << tailnode_indices.size() << ")";
  headnode_indices.clear();
  tailnode_indices.clear();

  return myapp;
}
