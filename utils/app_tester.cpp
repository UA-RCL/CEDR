#include <dlfcn.h>
#include <deque>
#include <iostream>
#include "dag_parse.hpp"
#include "zynq_runtime/header.hpp"

bool check_pred_comp(task_nodes *task) {
  for (int i = 0; i < task->pred_count; i++) {
    if (task->pred[i]->complete_flag == 0) {
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: ./app_tester <shared_object> <json_file>" << std::endl;
    return 1;
  }

  std::string shared_object_path(argv[1]);
  std::string shared_object_name = shared_object_path.substr(shared_object_path.find_last_of("/")+1);
  std::string json_path(argv[2]);

  std::map<std::string, void *> sharedObjectMap;

  void *dl_handle = dlopen(shared_object_path.c_str(), RTLD_LAZY);
  if (!dl_handle) {
    std::cerr << "Failed to open shared object file!" << std::endl << std::string(dlerror()) << std::endl;
    return 1;
  }
  sharedObjectMap[shared_object_name] = dl_handle;
  dag_app *myapp = parse_dag_file(json_path.c_str(), sharedObjectMap, false);

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

  // Perform an application instantiation similar to how the framework does
//  dag_app actual_app = *myapp;
//  myapp = &actual_app;

  std::deque<task_nodes *> ready_queue;

  for (int j = 0; j < myapp->task_count; j++) {
    task_nodes *tmp = &(myapp->head_node[j]);
    bool predecessors_complete = check_pred_comp(tmp);
    if ((tmp->complete_flag == 0) && (predecessors_complete) && (tmp->running_flag == 0)) {
      ready_queue.push_front(tmp);
    }
  }

  while (ready_queue.size() > 0) {
    task_nodes *task = ready_queue.back();
    ready_queue.pop_back();

//    std::vector<variable *> args = task->args;
    bound_safe_vector<variable *> args = task->args;
    void (*run_func)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                     void *, void *, void *, void *, void *, void *, void *, void *);
    *reinterpret_cast<void **>(&run_func) = task->run_func;

    if (args.size() > 20) {
      std::cerr << "Task " << std::string(task->task_name) << " has too many arguments in its dispatch call. "
                << "Please increase the maximum number of supported arguments and try running again.";
    } else {
      run_func(args.at(0)->heap_ptr, args.at(1)->heap_ptr, args.at(2)->heap_ptr, args.at(3)->heap_ptr,
               args.at(4)->heap_ptr, args.at(5)->heap_ptr, args.at(6)->heap_ptr, args.at(7)->heap_ptr,
               args.at(8)->heap_ptr, args.at(9)->heap_ptr, args.at(10)->heap_ptr, args.at(11)->heap_ptr,
               args.at(12)->heap_ptr, args.at(13)->heap_ptr, args.at(14)->heap_ptr, args.at(15)->heap_ptr,
               args.at(16)->heap_ptr, args.at(17)->heap_ptr, args.at(18)->heap_ptr, args.at(19)->heap_ptr);
    }

//    if (args.empty()) {
//      void (*run_func)();
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func();
//    } else if (args.size() == 1) {
//      void (*run_func)(void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0));
//    } else if (args.size() == 2) {
//      void (*run_func)(void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1));
//    } else if (args.size() == 3) {
//      void (*run_func)(void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2));
//    } else if (args.size() == 4) {
//      void (*run_func)(void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3));
//    } else if (args.size() == 5) {
//      void (*run_func)(void*, void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3), args.at(4));
//    } else if (args.size() == 6) {
//      void (*run_func)(void*, void*, void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3), args.at(4), args.at(5));
//    } else if (args.size() == 7) {
//      void (*run_func)(void*, void*, void*, void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3), args.at(4), args.at(5), args.at(6));
//    } else if (args.size() == 8) {
//      void (*run_func)(void*, void*, void*, void*, void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3), args.at(4), args.at(5), args.at(6), args.at(7));
//    } else if (args.size() == 9) {
//      void (*run_func)(void*, void*, void*, void*, void*, void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3), args.at(4), args.at(5), args.at(6), args.at(7), args.at(8));
//    } else if (args.size() == 10) {
//      void (*run_func)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3), args.at(4), args.at(5), args.at(6), args.at(7), args.at(8), args.at(9));
//    } else if (args.size() == 11) {
//      void (*run_func)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3), args.at(4), args.at(5), args.at(6), args.at(7), args.at(8), args.at(9), args.at(10));
//    } else if (args.size() == 12) {
//      void (*run_func)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3), args.at(4), args.at(5), args.at(6), args.at(7), args.at(8), args.at(9), args.at(10), args.at(11));
//    } else if (args.size() == 13) {
//      void (*run_func)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3), args.at(4), args.at(5), args.at(6), args.at(7), args.at(8), args.at(9), args.at(10), args.at(11), args.at(12));
//    } else if (args.size() == 14) {
//      void (*run_func)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3), args.at(4), args.at(5), args.at(6), args.at(7), args.at(8), args.at(9), args.at(10), args.at(11), args.at(12), args.at(13));
//    } else if (args.size() == 15) {
//      void (*run_func)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
//      *reinterpret_cast<void **>(&run_func) = task->run_func;
//      run_func(args.at(0), args.at(1), args.at(2), args.at(3), args.at(4), args.at(5), args.at(6), args.at(7), args.at(8), args.at(9), args.at(10), args.at(11), args.at(12), args.at(13), args.at(14));
//    } else {
//      std::cerr << "Task " << std::string(task->task_name) << " has too many arguments in its dispatch call. "
//                << "Please increase the maximum number of supported arguments and try running again.";
//    }

    task->complete_flag = 1;

    for (int j = 0; j < task->succ_count; j++) {
      if (check_pred_comp(task->succ[j])) {
        ready_queue.push_front(task->succ[j]);
      }
    }
  }
}