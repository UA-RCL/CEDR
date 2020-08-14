#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#include <sstream>
#include <vector>

#include "ipc.h"

#define USEC2NSEC 1000
int main(int argc, char **argv) {
  struct process_ipc_struct ipc_cmd;
  int option;
  char *dag_path = NULL;
  char *binary_path = NULL;
  int dflag = 0, sflag = 0;
  char cmd_s0_busy[] = CMD_SLOT0_BUSY;
  size_t dag_len;
  size_t bin_path_len;
  long long inj_period = 0;
  int inj_period_flag = 0;
  long long inj_duration = 0;
  int inj_dur_flag = 0;

  while ((option = getopt(argc, argv, "d:s:p:t:")) != -1) {
    switch (option) {
    case 'd':
      dag_path = optarg;
      dflag = 1;
      printf("DAG is %s\n", dag_path);

      break;
    case 's':
      binary_path = optarg;
      sflag = 1;
      printf("Binary path is %s\n", binary_path);
      break;
    case 't':
      inj_duration = atoi(optarg);
      inj_dur_flag = 1;
      printf("Time duration for injection is %lld \n", inj_duration);
      break;
    case 'p':
      inj_period = atoi(optarg);
      inj_period_flag = 1;
      printf("Injection period is %lld \n", inj_period);
      break;
    default:
      // printf("%s Option is incorrect\n", argv[optind-1]);
      printf("Usage DASH-sub -d \"<DAG.json>\" -s \"<path to shared object>\" [-p \"<period for injection>\" -t "
             "\"<duration of injection>\"] \n");
      return 1;
    }
  }

  // for (int index = optind; index < argc; index++){
  //	printf ("Non-option argument %s\n", argv[index]);
  //	printf("Usage DASH-sub -d \"<DAG.json>\" -s \"<path to shared object>\" \n");
  //	return 1;
  //}

  if ((dflag != 1) || (sflag != 1)) {
    printf("Failed Submission: Usage DASH-sub -d \"<DAG.json>\" -s \"<path to shared object>\" \n");
    return 1;
  }
  if (((inj_period_flag & inj_dur_flag) == 0) && ((inj_period_flag | inj_dur_flag) != 0)) {
    printf("Failed Submission: Usage DASH-sub -d \"<DAG.json>\" -s \"<path to shared object>\" [-p \"<period for "
           "injection>\" -t \"<duration of injection>\"] \n");
    printf("Failed Submission: -p (or -t) should be used with -t (or -p)\n");
    return 1;
  }
  // itoa(strlen(dag_path),dag_len,10);
  // itoa(strlen(binary_path),bin_path_len,10);

  if ((inj_period_flag == 0) && (inj_dur_flag == 0)) {

    dag_len = strlen(dag_path);
    bin_path_len = strlen(binary_path);
    ipc_cmd.message_len = sizeof(CMD_SLOT0_BUSY) + sizeof(size_t) + dag_len + 1 + sizeof(size_t) + bin_path_len + 1;
    ipc_cmd.message = (char *)malloc((ipc_cmd.message_len) * sizeof(char));
    memcpy(ipc_cmd.message, cmd_s0_busy, sizeof(CMD_SLOT0_BUSY));
    memcpy(&(ipc_cmd.message[sizeof(CMD_SLOT0_BUSY)]), (&dag_len), sizeof(size_t));
    memcpy(&(ipc_cmd.message[sizeof(CMD_SLOT0_BUSY) + sizeof(size_t)]), dag_path, dag_len + 1);
    memcpy(&(ipc_cmd.message[sizeof(CMD_SLOT0_BUSY) + sizeof(size_t) + dag_len + 1]), (&bin_path_len), sizeof(size_t));
    memcpy(&(ipc_cmd.message[sizeof(CMD_SLOT0_BUSY) + sizeof(size_t) + dag_len + 1 + sizeof(size_t)]), binary_path,
           bin_path_len + 1);

    initialize_sh_mem_client(&ipc_cmd);
    // client_submission(&ipc_cmd);
    free(ipc_cmd.message);

  } else {
    int expected_instances = ceil(inj_duration / inj_period);
    int inj_instances = 0;
    struct timespec sleep_time;
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = inj_period * USEC2NSEC;
    std::string folder_name;
    {
      std::stringstream test(binary_path);
      std::string segment;
      std::vector<std::string> seglist;
      while (std::getline(test, segment, '/')) {
        seglist.push_back(segment);
      }
      folder_name = seglist.at(seglist.size() - 1);
    }

    while (inj_instances < expected_instances) {

      std::string tmp_fld("/tmp/");
      tmp_fld = tmp_fld + folder_name + std::to_string(inj_instances) + "/";
      std::string cmd = "cp -rf ";
      cmd = cmd + binary_path + "  " + tmp_fld;
      const char *command = cmd.c_str();
      system(command);
      std::string new_binary_path = tmp_fld;

      dag_len = strlen(dag_path);
      bin_path_len = strlen(new_binary_path.c_str());
      ipc_cmd.message_len = sizeof(CMD_SLOT0_BUSY) + sizeof(size_t) + dag_len + 1 + sizeof(size_t) + bin_path_len + 1;
      ipc_cmd.message = (char *)malloc((ipc_cmd.message_len) * sizeof(char));
      memcpy(ipc_cmd.message, cmd_s0_busy, sizeof(CMD_SLOT0_BUSY));
      memcpy(&(ipc_cmd.message[sizeof(CMD_SLOT0_BUSY)]), (&dag_len), sizeof(size_t));
      memcpy(&(ipc_cmd.message[sizeof(CMD_SLOT0_BUSY) + sizeof(size_t)]), dag_path, dag_len + 1);
      memcpy(&(ipc_cmd.message[sizeof(CMD_SLOT0_BUSY) + sizeof(size_t) + dag_len + 1]), (&bin_path_len),
             sizeof(size_t));
      memcpy(&(ipc_cmd.message[sizeof(CMD_SLOT0_BUSY) + sizeof(size_t) + dag_len + 1 + sizeof(size_t)]),
             new_binary_path.c_str(), bin_path_len + 1);

      initialize_sh_mem_client(&ipc_cmd);
      // client_submission(&ipc_cmd);
      free(ipc_cmd.message);
      nanosleep(&sleep_time, nullptr);
      inj_instances++;
    }
  }

  return 0;
}
