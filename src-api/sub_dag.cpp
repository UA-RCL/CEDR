#include "ipc.h"
#include <fcntl.h>
#include <pthread.h>
#include <sys/types.h>
#include <cstdlib>
#include <cstring>

#include <jarro2783/cxxopts.hpp>

#define member_size(type, member) sizeof((type){}.member)

int main(int argc, char **argv) {
  struct process_ipc_struct ipc_cmd {};
  char cmd_s0_busy[] = CMD_SLOT0_BUSY;

  //cpu_set_t scheduler_affinity;
  //CPU_ZERO(&scheduler_affinity);
  //CPU_SET(0, &scheduler_affinity);

  //pthread_t current_thread = pthread_self();
  //pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &scheduler_affinity);

  //  struct sched_param main_thread;
  //  main_thread.sched_priority = 99;  // If using SCHED_RR, set this
  //  pthread_setschedparam(current_thread, SCHED_RR, &main_thread);

  cxxopts::Options options("sub_dag", "A helper program for submitting applications to a daemon-based CEDR process");

  // clang-format off
  options.add_options()   // TODO: Change expected input argument printing
  ("a,app-shared-obj", "Specify a list of application shared object files to invoke", cxxopts::value<std::vector<std::string>>())
  ("n,num-instances", "For each application, specify the number of instances to invoke", cxxopts::value<std::vector<uint64_t>>())
  ("p,periodicity", "Specifies a periodic injection for instances", cxxopts::value<std::vector<uint64_t>>())
  ("h,help","Print help")
  ;
  // clang-format on

  try {
    const auto args = options.parse(argc, argv);

    if (args.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
    }

    const std::vector<std::string> soPaths = args["app-shared-obj"].as<std::vector<std::string>>();
    const std::vector<uint64_t> numInstances = args["num-instances"].as<std::vector<uint64_t>>();
    std::vector<uint64_t> period = std::vector<uint64_t>();

    if (soPaths.empty()) {
      std::cerr << "[sub_dag] No shared object files were specified!" << std::endl;
      std::cerr << options.help() << std::endl;
      return 1;
    }

    if (soPaths.size() > IPC_MAX_APPS) {
      std::cerr << "[sub_dag] Too many shared object files were specified!" << std::endl;
      std::cerr << "[sub_dag] Adjust the maximum number of shared object files in ipc.h and then recompile" << std::endl;
      return 1;
    }

    if (soPaths.size() != numInstances.size()) {
      std::cerr << "[sub_dag] Number of instances specified does not match number of shared object files submitted!" << std::endl;
      std::cerr << options.help() << std::endl;
      return 1;
    }

    for (const auto &soPath : soPaths) {
      if (soPath.size() > member_size(struct ipc_message_struct, path_to_so)) {
        std::cerr << "[sub_dag] Unable to submit shared object: " << soPath << std::endl;
        std::cerr << "[sub_dag] Shared object path length (" << soPath.size() << ") exceeds " << member_size(struct ipc_message_struct, path_to_so) << " characters" << std::endl;
        std::cerr << "[sub_dag] Adjust the maximum size in ipc.h and then recompile" << std::endl;
        return 1;
      }
    }

    if (args.count("periodicity") != 0) {
      period = args["periodicity"].as<std::vector<uint64_t>>();
    } else {
      std::cout << "[sub_dag] No period specified. Made all periods into zeros" << std::endl;
    }

    // Allocate one message struct
    ipc_cmd.message = (struct ipc_message_struct *)calloc(1, sizeof(struct ipc_message_struct));

    // Eventually, when we copy this struct into the shared memory buffer, we want CEDR to read and realize there's a
    // DAG
    memcpy(&(ipc_cmd.message->CMD), cmd_s0_busy, sizeof(CMD_SLOT0_BUSY));

    // Set the parameters of our message
    for (auto i = 0; i < soPaths.size(); i++) {
      const auto &soPath = soPaths.at(i);
      const auto numInstance = numInstances.at(i);
      const auto frequency = !period.empty() ? period.at(i) : 0;
      memcpy(ipc_cmd.message->path_to_so[i], soPath.c_str(), soPath.size());
      ipc_cmd.message->num_instances[i] = numInstance;
      ipc_cmd.message->periodicity[i] = frequency;
    }
    initialize_sh_mem_client(&ipc_cmd);
    // client_submission(&ipc_cmd);
    free(ipc_cmd.message);

    return 0;
  } catch (cxxopts::option_not_exists_exception &e) {
    std::cerr << "[sub_dag] Unrecognized option specified: " << std::string(e.what()) << std::endl;
    return 1;
  }
}
